# Internal UniMVU variant with mixed-modality-safe unified training.

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers import AutoModelForCausalLM

from libs.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

from ._unimvu_base import (
    UniMVUQwen2Config,
    UniMVUQwen2Model,
    UniMVUQwen2ForCausalLM,
)

def _parse_modality_dim_map(raw_value) -> Dict[str, int]:
    """
    Normalize the modality->input-dimension mapping.

    Accepts:
        - dict[str, int]
        - list/tuple of (name, dim)
        - comma separated string "video:1024,audio:768"
    """
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return {str(k): int(v) for k, v in raw_value.items()}
    if isinstance(raw_value, (list, tuple)):
        parsed = {}
        for item in raw_value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, dim = item
                parsed[str(name)] = int(dim)
            else:
                raise ValueError(f"Unsupported modality_input_dims element: {item!r}")
        return parsed
    if isinstance(raw_value, str):
        parsed: Dict[str, int] = {}
        for chunk in raw_value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"Invalid modality_input_dims entry '{chunk}'. Expected 'name:dim'.")
            name, dim = chunk.split(":", maxsplit=1)
            parsed[name.strip()] = int(dim.strip())
        return parsed
    raise ValueError(f"Unsupported modality_input_dims type: {type(raw_value)}")


class UniMVUUnifiedConfig(UniMVUQwen2Config):
    """Configuration alias enabling unified mixed-dataset training."""

    model_type = "unimvu_uni"

    def __init__(self, model_config=None, **kwargs):
        super(UniMVUUnifiedConfig, self).__init__(model_config=model_config, **kwargs)
        self.default_input_dim = getattr(self, "default_input_dim", getattr(self, "input_dim", None))
        if self.default_input_dim is None:
            raise ValueError(
                "UniMVUUnifiedConfig requires `input_dim` or `default_input_dim` \
                for different modality input dimensions.")

        raw_map = getattr(self, "modality_input_dims", None)
        self.modality_input_dims = _parse_modality_dim_map(raw_map)


class UniMVUUnifiedForCausalLM(UniMVUQwen2ForCausalLM):
    """
    Unified UniMVU variant that pads missing modalities with masked placeholders
    so all modality branches stay active inside ZeRO Stage-2.
    """

    config_class = UniMVUUnifiedConfig

    def __init__(self, config: UniMVUUnifiedConfig):
        super().__init__(config)
        default_dim = config.default_input_dim
        modality_dims = dict(config.modality_input_dims)

        if isinstance(config.support_modalities, str):
            supported = [m.strip() for m in config.support_modalities.split(",") if m.strip()]
        else:
            supported = list(config.support_modalities) if config.support_modalities else []

        projector_keys = set(supported)
        projector_keys.discard("text")
        if "default" not in modality_dims:
            modality_dims["default"] = default_dim

        self.modality_projectors = nn.ModuleDict()
        for key in sorted(projector_keys):
            in_dim = modality_dims.get(key, default_dim)
            self.modality_projectors[key] = nn.Linear(in_dim, config.hidden_size, bias=False)
        if "default" not in self.modality_projectors:
            self.modality_projectors["default"] = nn.Linear(modality_dims["default"], config.hidden_size, bias=False)

        for projector in self.modality_projectors.values():
            self._init_weights(projector)

        self.all_modalities = [
            modality
            for modality in self.supported_modalities
            if modality not in {"video", "text"}
        ]
        self._stage2_placeholder_info: List[Set[str]] = []
        self.modality_type_ids = {
            "video": 1,
            "audio": 2,
            "3d_feature": 3,
            "dense_video": 4,
            "exo_random": 5,
        }
        self.modality_follow_combine_method = ["3d_feature", "exo_random"]

    def _expand_to_list(
        self,
        value: Optional[Union[Sequence, torch.Tensor, int, float]],
        batch_size: int,
        convert_to_int: bool = False,
    ) -> List[Optional[Union[torch.Tensor, float, int]]]:
        if value is None:
            return [None] * batch_size

        if isinstance(value, (list, tuple)):
            if len(value) == batch_size:
                result = list(value)
            elif len(value) == 1:
                result = [value[0]] * batch_size
            else:
                raise ValueError(
                    f"Cannot broadcast list of length {len(value)} to batch size {batch_size}."
                )
        elif torch.is_tensor(value):
            if value.ndim == 0:
                result = [value.item()] * batch_size
            elif value.shape[0] == batch_size:
                result = [value[i] for i in range(batch_size)]
            else:
                raise ValueError(
                    f"Cannot broadcast tensor of shape {tuple(value.shape)} to batch size {batch_size}."
                )
        else:
            result = [value] * batch_size

        if convert_to_int:
            converted: List[Optional[int]] = []
            for item in result:
                if item is None:
                    converted.append(None)
                elif torch.is_tensor(item):
                    converted.append(int(item.item()))
                else:
                    converted.append(int(item))
            return converted
        return result  # type: ignore[return-value]

    def _slice_frames(
        self,
        sample: Optional[torch.Tensor],
        frame_count: Optional[int],
    ) -> Optional[torch.Tensor]:
        if sample is None or frame_count is None:
            return sample
        if not torch.is_tensor(sample):
            sample = torch.as_tensor(sample)

        if sample.ndim >= 2:
            if sample.shape[1] >= frame_count:
                return sample[:, :frame_count, ...].contiguous()
            if sample.shape[0] >= frame_count:
                return sample[:frame_count, ...].contiguous()
        return sample.contiguous()

    def _split_feature_tensor(
        self,
        raw: Optional[Union[Sequence[torch.Tensor], torch.Tensor]],
        frame_counts: Sequence[Optional[int]],
        batch_size: int,
    ) -> List[Optional[torch.Tensor]]:
        if raw is None:
            return [None] * batch_size

        outputs: List[Optional[torch.Tensor]] = []

        if isinstance(raw, (list, tuple)):
            if len(raw) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} modality entries, but received {len(raw)}."
                )
            for idx, item in enumerate(raw):
                outputs.append(self._slice_frames(item, frame_counts[idx]))
            return outputs

        if torch.is_tensor(raw):
            if raw.shape[0] != batch_size:
                raise ValueError(
                    f"Expected tensor batch dimension {batch_size}, but received {raw.shape[0]}."
                )
            for idx in range(batch_size):
                outputs.append(self._slice_frames(raw[idx], frame_counts[idx]))
            return outputs

        raise TypeError(f"Unsupported modality payload type: {type(raw)}")

    def _build_frame_lookup(
        self,
        feat_frame_nums,
        batch_size: int,
    ) -> Dict[str, List[Optional[int]]]:
        if isinstance(feat_frame_nums, dict):
            lookup = {
                key: self._expand_to_list(val, batch_size, convert_to_int=True)
                for key, val in feat_frame_nums.items()
            }
        else:
            lookup = {
                "__default__": self._expand_to_list(
                    feat_frame_nums, batch_size, convert_to_int=True
                )
            }

        lookup.setdefault("__default__", [None] * batch_size)
        return lookup

    def _ensure_last_dim(self, tensor: torch.Tensor, expected_dim: int, modality: str) -> torch.Tensor:
        if tensor.shape[-1] == expected_dim:
            return tensor.contiguous()

        if tensor.ndim == 4:
            if tensor.shape[0] == expected_dim:
                return tensor.permute(1, 2, 3, 0).contiguous()
            if tensor.shape[1] == expected_dim:
                return tensor.permute(0, 2, 3, 1).contiguous()

        if tensor.ndim == 3:
            if tensor.shape[0] == expected_dim:
                return tensor.permute(1, 2, 0).contiguous()
            if tensor.shape[1] == expected_dim:
                return tensor.permute(0, 2, 1).contiguous()

        if tensor.ndim == 2 and tensor.shape[0] == expected_dim:
            return tensor.transpose(0, 1).contiguous()

        raise ValueError(
            f"Unable to align modality '{modality}' features of shape {tuple(tensor.shape)} "
            f"to expected dimension {expected_dim}."
        )

    def _normalize_fast_feature_collection(
        self,
        video_feats,
        modalities_per_sample: Sequence[Sequence[str]],
        feat_frame_nums,
        batch_size: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert incoming auxiliary/fast features into a per-sample modality map.
        """
        normalized = [dict() for _ in range(batch_size)]
        frame_lookup = self._build_frame_lookup(feat_frame_nums, batch_size)

        def assign(idx: int, modality: str, tensor: Optional[torch.Tensor]):
            if tensor is None:
                return
            frames = frame_lookup.get(modality) or frame_lookup["__default__"]
            sliced = self._slice_frames(tensor, frames[idx] if idx < len(frames) else None)
            if sliced is not None:
                normalized[idx][modality] = sliced

        if video_feats is not None:
            if isinstance(video_feats, dict):
                for modality_name, payload in video_feats.items():
                    per_sample = self._split_feature_tensor(
                        payload,
                        frame_lookup.get(modality_name, frame_lookup["__default__"]),
                        batch_size,
                    )
                    for idx, tensor in enumerate(per_sample):
                        assign(idx, modality_name, tensor)
            elif isinstance(video_feats, (list, tuple)):
                if len(video_feats) != batch_size:
                    raise ValueError(
                        f"Expected {batch_size} fast feature entries, but received {len(video_feats)}."
                    )
                for idx, sample_entry in enumerate(video_feats):
                    if sample_entry is None:
                        continue
                    if isinstance(sample_entry, dict):
                        for modality_name, tensor in sample_entry.items():
                            assign(idx, modality_name, tensor)
                        continue
                    if isinstance(sample_entry, (list, tuple)):
                        fallback_modalities = list(modalities_per_sample[idx]) if modalities_per_sample else []
                        if not fallback_modalities:
                            fallback_modalities = ["video"]
                        for mod_idx, tensor in enumerate(sample_entry):
                            modality_name = (
                                fallback_modalities[mod_idx]
                                if mod_idx < len(fallback_modalities)
                                else f"aux_{mod_idx}"
                            )
                            assign(idx, modality_name, tensor)
                        continue
                    assign(idx, "video", sample_entry)
            # Single shared tensor: broadcast to the first auxiliary modality per sample.
            # TODO: add support for multiple auxiliary modalities
            elif torch.is_tensor(video_feats):
                if video_feats.shape[0] == batch_size:
                    for idx in range(batch_size):
                        if not self.training and batch_size == 1:
                            modalities_per_sample = [modalities_per_sample]
                        aux_modalities = [m for m in self.all_modalities if m in modalities_per_sample[idx]]
                        assign(idx, aux_modalities[0], video_feats[idx])

                else:
                    per_sample = self._split_feature_tensor(
                        video_feats, frame_lookup["__default__"], batch_size
                    )
                    for idx, tensor in enumerate(per_sample):
                        assign(idx, "video", tensor)
            else:
                raise TypeError(f"Unsupported modality payload type: {type(video_feats)}")

        placeholder_info = []
        modalities_with_placeholders = []
        if self.all_modalities:
            for idx in range(batch_size):
                placeholders = set()
                sample_features = normalized[idx]
                sample_modalities = list(modalities_per_sample[idx]) if modalities_per_sample else []
                for modality in self.all_modalities:
                    if modality in sample_features:
                        if modality not in sample_modalities:
                            sample_modalities.append(modality)
                        continue
                    zero_feat = torch.zeros(
                        (1, self.modality_projectors[modality].in_features),
                        dtype=self.modality_projectors[modality].weight.dtype,
                        device=self.modality_projectors[modality].weight.device,
                    )
                    sample_features[modality] = zero_feat
                    placeholders.add(modality)
                    if modality not in sample_modalities:
                        sample_modalities.append(modality)
                placeholder_info.append(placeholders)
                modalities_with_placeholders.append(sample_modalities)
        else:
            placeholder_info = [set() for _ in range(batch_size)]
            modalities_with_placeholders = [
                list(modalities_per_sample[idx]) if modalities_per_sample else []
                for idx in range(batch_size)
            ]

        self.placeholder_info = placeholder_info
        self.modalities_with_placeholders = modalities_with_placeholders
        return normalized

    def _process_fast_stream_placeholder(
        self,
        modality: str,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = self._ensure_last_dim(feature, self.modality_projectors[modality].in_features, modality)
        return self.modality_projectors[modality](feature)

    def _process_fast_stream(
        self,
        modality: str,
        feature: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
        scaled_shape,
        feat_combine_method: str,
    ) -> torch.Tensor:
        feature = self._ensure_last_dim(feature, self.modality_projectors[modality].in_features, modality)

        projector_dtype = self.modality_projectors[modality].weight.dtype
        if feature.device != target_device or feature.dtype != projector_dtype:
            feature = feature.to(
                device=target_device,
                dtype=projector_dtype,
                non_blocking=True,
            )
        else:
            feature = feature.contiguous()

        feature = self.modality_projectors[modality](feature)
        if feature.dtype != target_dtype:
            feature = feature.to(dtype=target_dtype, non_blocking=True)

        if modality in self.modality_follow_combine_method and feature.ndim >= 4 and scaled_shape is not None:
            if feat_combine_method == "add":
                if feature.shape[1] != scaled_shape[0] or feature.shape[2] != scaled_shape[1]:
                    feature = self.image_shape_alignment(feature, scaled_shape)
                feature = rearrange(feature, "t h w d -> t (h w) d")
                feature = self.post_processing_of_image_feature([feature], [0])[0]
                # feature = rearrange(feature, "t h w d -> (t h w) d")
            elif feat_combine_method == "concat":
                feature = rearrange(feature, "t h w d -> (t h w) d")
            else:
                raise ValueError(f"Invalid feat_combine_method: {feat_combine_method}")
        else:
            if feature.ndim >= 4:
                feature = rearrange(feature, "t h w d -> (t h w) d")
            elif feature.ndim == 3:
                feature = feature.reshape(-1, feature.shape[-1])
            elif feature.ndim == 2:
                feature = feature
            else:
                raise ValueError(f"Unsupported feature dimension: {feature.ndim}")

        prepend_token = modality != "video"
        if prepend_token:
            modality_token = self.modality_tokens.to(device=target_device, dtype=target_dtype)
            stream = torch.cat([modality_token, feature], dim=0)
        else:
            stream = feature

        return stream


    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        video_feats,
        video_feat_fps=None,
        feat_frame_nums=None,
        question_ids=None,
        question_lens=None,
        images=None,
        image_sizes=None,
        modalities=None,
        video_metas=None,
    ):
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif attention_mask is not None:
            batch_size = attention_mask.shape[0]
        elif isinstance(video_feats, list):
            batch_size = len(video_feats)
        elif isinstance(images, list):
            batch_size = len(images)
        else:
            batch_size = 1

        video_tower = self.get_video_tower()
        vision_tower = self.get_vision_tower()

        if (video_tower is None and vision_tower is None) or \
            (video_feats is None and images is None) or \
            input_ids.shape[1] == 1: # this could be used for the inference
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        self.use_slow_feat_before_mlp = True

        if images is not None and image_sizes is not None and (-1 not in image_sizes):
            image_features, video_idx_in_batch, image_features_before_mlp, scaled_shape = self.prepare_image_features(
                images,
                image_sizes,
                modalities,
                return_feat_before_mlp=True,
            )
        else:
            if image_sizes is not None and sum(image_sizes) != -len(image_sizes):
                raise ValueError(
                    f"Invalid image_sizes: expected all elements to be -1, but got {image_sizes}"
                )
            image_features = images
            video_idx_in_batch: List[int] = []
            for idx, sample_modalities in enumerate(modalities):
                if "video" in sample_modalities:
                    video_idx_in_batch.append(idx)
            scaled_shape = None

        if images is not None:
            image_features = self.post_processing_of_image_feature(
                image_features,
                video_idx_in_batch,
            )

        fast_features_per_sample = self._normalize_fast_feature_collection(
            video_feats=video_feats,
            modalities_per_sample=modalities,
            feat_frame_nums=feat_frame_nums,
            batch_size=batch_size,
        )

        placeholder_info = self.placeholder_info or [set() for _ in range(batch_size)]
        modalities_with_placeholders = self.modalities_with_placeholders

        assembled_features: List[torch.Tensor] = []
        feature_token_types: List[torch.Tensor] = []
        placeholder_feature_masks: List[torch.Tensor] = []

        for batch_idx in range(batch_size):
            sample_modalities = modalities_with_placeholders[batch_idx]
            sample_placeholders = placeholder_info[batch_idx]
            curr_image_feat = None
            if image_features is not None:
                curr_image_feat = image_features[batch_idx]

            stream_tokens = []
            stream_types = []
            stream_placeholder_masks = []

            if curr_image_feat is not None:
                target_device = curr_image_feat.device
                target_dtype = curr_image_feat.dtype
                video_control_token = self.modality_tokens.to(device=target_device, dtype=target_dtype)
                video_stream = torch.cat([video_control_token, curr_image_feat], dim=0)
                stream_tokens.append(video_stream)
                stream_types.append(
                    torch.full(
                        (video_stream.shape[0],),
                        self.modality_type_ids["video"],
                        device=target_device,
                        dtype=torch.long,
                    )
                )
                stream_placeholder_masks.append(
                    torch.zeros(video_stream.shape[0], dtype=torch.bool, device=target_device)
                )

            sample_fast_map = fast_features_per_sample[batch_idx]

            for modality in sample_modalities:
                if modality == "video":
                    continue
                if modality not in self.supported_modalities:
                    raise ValueError(f"Unsupported modality: {modality}")
                feat_tensor = sample_fast_map.get(modality)
                if feat_tensor is None:
                    continue
                if modality in sample_placeholders:
                    processed_stream = self._process_fast_stream_placeholder(
                        modality=modality,
                        feature=feat_tensor,
                    )
                    stream_tokens.append(processed_stream)
                    stream_types.append(
                        torch.full(
                            (feat_tensor.shape[0],),
                            -1,
                            device=processed_stream.device,
                            dtype=torch.long,
                        )
                    )
                    stream_placeholder_masks.append(
                        torch.ones(
                            processed_stream.shape[0],
                            dtype=torch.bool,
                            device=processed_stream.device,
                        )
                    )
                    continue

                processed_stream = self._process_fast_stream(
                    modality=modality,
                    feature=feat_tensor,
                    target_device=target_device,
                    target_dtype=target_dtype,
                    scaled_shape=scaled_shape,
                    feat_combine_method=getattr(self.config, "feat_combine_method", "concat"),
                )
                stream_tokens.append(processed_stream)
                stream_types.append(
                    torch.full(
                        (processed_stream.shape[0],),
                        self.modality_type_ids.get(modality, self.modality_type_ids["video"]),
                        device=processed_stream.device,
                        dtype=torch.long,
                    )
                )
                stream_placeholder_masks.append(
                    torch.zeros(
                        processed_stream.shape[0],
                        dtype=torch.bool,
                        device=processed_stream.device,
                    )
                )

            if not stream_tokens:
                empty = torch.zeros(
                    (0, self.config.hidden_size),
                    device=self.modality_tokens.device,
                    dtype=self.modality_tokens.dtype,
                )
                assembled_features.append(empty)
                feature_token_types.append(
                    torch.zeros(0, device=empty.device, dtype=torch.long)
                )
                placeholder_feature_masks.append(
                    torch.zeros(0, dtype=torch.bool, device=empty.device)
                )
                continue

            combined_feat = torch.cat(stream_tokens, dim=0)
            combined_types = torch.cat(stream_types, dim=0)
            combined_placeholder_mask = torch.cat(stream_placeholder_masks, dim=0)
            assembled_features.append(combined_feat)
            feature_token_types.append(combined_types)
            placeholder_feature_masks.append(combined_placeholder_mask)

        new_frame_num = [feat.shape[0] for feat in assembled_features]

        orig_attention_mask = attention_mask
        (
            new_input_embeds,
            new_labels,
            attention_mask,
            position_ids,
        ) = self._assemble_text_and_feature_embeddings_stage2(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            video_features=assembled_features,
            new_frame_num=new_frame_num,
            placeholder_feature_masks=placeholder_feature_masks,
        )

        if self.training and new_input_embeds is not None and not new_input_embeds.requires_grad:
            new_input_embeds.requires_grad_(True)

        modality_indices = self._build_modality_indices(
            input_ids=input_ids,
            attention_mask=orig_attention_mask,
            feature_token_types=feature_token_types,
        )
        self.latest_modality_indices = modality_indices

        new_input_embeds = self.modality_special_token_aggregator(
            new_input_embeds,
            modality_indices,
        )

        new_input_embeds = self.modality_aggregator(
            hidden_states=new_input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_indices=modality_indices,
        )

        feat_combine_method = getattr(self.config, "feat_combine_method", "concat")
        needs_add_fusion = False
        if feat_combine_method == "add":
            for b in range(len(modalities)):
                if any(modality in modalities[b] for modality in self.modality_follow_combine_method):
                    needs_add_fusion = True
                    break

        if needs_add_fusion:
            B, _, H = new_input_embeds.shape
            keep_lengths = torch.zeros(B, dtype=torch.long, device=attention_mask.device)

            for b in range(B):
                vid_idx = torch.nonzero(
                    modality_indices[b] == self.modality_type_ids["video"], as_tuple=False
                ).flatten()
                for modality in self.modality_follow_combine_method:
                    modality_idx = torch.nonzero(
                        modality_indices[b] == self.modality_type_ids[modality],as_tuple=False,
                    ).flatten()
                    if vid_idx.numel() > 0 and modality_idx.numel() > 0:
                        if vid_idx.numel() != modality_idx.numel():
                            raise ValueError(
                                f"Fusion mismatch: video tokens ({vid_idx.numel()}) != {modality} tokens ({modality_idx.numel()})"
                            )
                        new_input_embeds[b, vid_idx] += new_input_embeds[b, modality_idx]
                        attention_mask[b, modality_idx] = 0
                        modality_indices[b, modality_idx] = -1
                    keep_lengths[b] = attention_mask[b].sum()

            max_len = int(keep_lengths.max().item()) if B > 0 else 0
            out_embeds = new_input_embeds.new_zeros((B, max_len, H))
            out_masks = attention_mask.new_zeros((B, max_len))
            out_mods = modality_indices.new_full((B, max_len), -1)
            out_pos = position_ids.new_zeros((B, max_len)) if position_ids is not None else None
            out_labels = new_labels.new_full((B, max_len), -100) if new_labels is not None else None

            for b in range(B):
                keep = attention_mask[b].bool()
                Lb = int(keep_lengths[b].item())
                if Lb == 0:
                    continue
                out_embeds[b, :Lb] = new_input_embeds[b][keep]
                out_masks[b, :Lb] = attention_mask[b][keep]
                out_mods[b, :Lb] = modality_indices[b][keep]
                if out_pos is not None and position_ids is not None:
                    out_pos[b, :Lb] = position_ids[b][keep]
                if out_labels is not None and new_labels is not None:
                    out_labels[b, :Lb] = new_labels[b][keep]

            new_input_embeds = out_embeds
            attention_mask = out_masks
            modality_indices = out_mods
            if out_pos is not None:
                position_ids = out_pos
            if out_labels is not None:
                new_labels = out_labels

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, modality_indices

    # ------------------------------------------------------------------ #
    # Custom embedding assembly with placeholder masking
    # ------------------------------------------------------------------ #
    def _assemble_text_and_feature_embeddings_stage2(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        video_features: Sequence[torch.Tensor],
        new_frame_num: Sequence[int],
        placeholder_feature_masks: Sequence[torch.Tensor],
    ):
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []
        feature_token_masks: List[torch.Tensor] = []
        placeholder_token_masks: List[torch.Tensor] = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_image_features = video_features[cur_image_idx]
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]],
                    dim=0,
                ).to(self.device)
                cur_labels_seq = labels[batch_idx]
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(cur_labels_seq)
                feature_token_masks.append(
                    torch.zeros(
                        cur_input_embeds.shape[0],
                        dtype=torch.bool,
                        device=cur_input_embeds.device,
                    )
                )
                placeholder_token_masks.append(
                    torch.zeros(
                        cur_input_embeds.shape[0],
                        dtype=torch.bool,
                        device=cur_input_embeds.device,
                    )
                )
                cur_image_idx += 1
                continue

            cur_labels_seq = labels[batch_idx]
            image_token_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_token_indices = [-1] + image_token_positions + [cur_input_ids.shape[0]]
            cur_input_ids_noim = [
                cur_input_ids[start + 1 : end]
                for start, end in zip(image_token_indices[:-1], image_token_indices[1:])
            ]
            cur_labels_noim = [
                cur_labels_seq[start + 1 : end]
                for start, end in zip(image_token_indices[:-1], image_token_indices[1:])
            ]
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

            cur_new_input_embeds: List[torch.Tensor] = []
            cur_new_labels: List[torch.Tensor] = []
            cur_feature_masks: List[torch.Tensor] = []
            cur_placeholder_masks: List[torch.Tensor] = []

            for i in range(num_images + 1):
                text_embed = cur_input_embeds_no_im[i]
                cur_new_input_embeds.append(text_embed)
                cur_new_labels.append(cur_labels_noim[i])
                zeros_mask = torch.zeros(
                    text_embed.shape[0],
                    dtype=torch.bool,
                    device=text_embed.device,
                )
                cur_feature_masks.append(zeros_mask)
                cur_placeholder_masks.append(zeros_mask)
                if i < num_images:
                    cur_image_features = video_features[cur_image_idx]
                    cur_feature_len = new_frame_num[cur_image_idx]
                    cur_image_features = cur_image_features[:cur_feature_len]
                    if cur_image_features.ndim == 3:
                        cur_image_features = cur_image_features.view(-1, cur_image_features.shape[-1])

                    placeholder_mask = placeholder_feature_masks[cur_image_idx]
                    placeholder_mask = placeholder_mask[:cur_image_features.shape[0]]
                    cur_image_idx += 1

                    cur_feature_mask = torch.ones(
                        cur_image_features.shape[0],
                        dtype=torch.bool,
                        device=cur_image_features.device,
                    )
                    cur_feature_mask = cur_feature_mask & (~placeholder_mask)

                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels_seq.device,
                            dtype=cur_labels_seq.dtype,
                        )
                    )
                    cur_feature_masks.append(cur_feature_mask)
                    cur_placeholder_masks.append(placeholder_mask.to(device=cur_image_features.device))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_feature_masks = [x.to(self.device) for x in cur_feature_masks]
            cur_placeholder_masks = [x.to(self.device) for x in cur_placeholder_masks]

            cur_new_input_embeds_tensor = torch.cat(cur_new_input_embeds, dim=0)
            cur_new_labels_tensor = torch.cat(cur_new_labels, dim=0)
            cur_feature_mask_tensor = torch.cat(cur_feature_masks, dim=0)
            cur_placeholder_mask_tensor = torch.cat(cur_placeholder_masks, dim=0)

            new_input_embeds.append(cur_new_input_embeds_tensor)
            new_labels.append(cur_new_labels_tensor)
            feature_token_masks.append(cur_feature_mask_tensor)
            placeholder_token_masks.append(cur_placeholder_mask_tensor)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            truncated_input_embeds: List[torch.Tensor] = []
            truncated_labels: List[torch.Tensor] = []
            truncated_masks: List[torch.Tensor] = []
            truncated_placeholder_masks: List[torch.Tensor] = []
            for embeds, label_seq, feature_mask, placeholder_mask in zip(
                new_input_embeds, new_labels, feature_token_masks, placeholder_token_masks
            ):
                seq_len = embeds.shape[0]
                if seq_len <= tokenizer_model_max_length:
                    truncated_input_embeds.append(embeds)
                    truncated_labels.append(label_seq)
                    truncated_masks.append(feature_mask)
                    truncated_placeholder_masks.append(placeholder_mask)
                    continue

                text_mask = ~(feature_mask | placeholder_mask)
                text_indices = torch.nonzero(text_mask, as_tuple=False).flatten()
                feature_indices = torch.nonzero(feature_mask, as_tuple=False).flatten()
                # placeholder_indices = torch.nonzero(placeholder_mask, as_tuple=False).flatten()
                text_len = text_indices.numel()

                keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=embeds.device)

                if text_len >= tokenizer_model_max_length:
                    keep_text = text_indices[:tokenizer_model_max_length]
                    if keep_text.numel() > 0:
                        keep_mask[keep_text] = True
                else:
                    keep_mask[text_indices] = True
                    remaining_slots = tokenizer_model_max_length - text_len
                    if remaining_slots > 0 and feature_indices.numel() > 0:
                        keep_features = feature_indices[:remaining_slots]
                        if keep_features.numel() > 0:
                            keep_mask[keep_features] = True
                    # ensure placeholders never take priority

                truncated_input_embeds.append(embeds[keep_mask])
                keep_mask_labels = keep_mask.to(label_seq.device)
                truncated_labels.append(label_seq[keep_mask_labels])
                truncated_masks.append(feature_mask[keep_mask])
                truncated_placeholder_masks.append(placeholder_mask[keep_mask])

            new_input_embeds = truncated_input_embeds
            new_labels = truncated_labels
            feature_token_masks = truncated_masks
            placeholder_token_masks = truncated_placeholder_masks

        (
            input_embeds_tensor,
            labels_padded,
            attention_mask_padded,
            position_ids_padded,
        ) = self._pad_and_stack_new_sequences(
            new_input_embeds,
            new_labels,
            attention_mask,
            position_ids,
        )

        placeholder_mask_padded = self._pad_placeholder_masks(
            placeholder_token_masks,
            attention_mask_padded,
        )

        attention_mask_padded = attention_mask_padded & (~placeholder_mask_padded)

        if _labels is None:
            labels_out = None
        else:
            labels_out = labels_padded

        if _attention_mask is None:
            attention_mask_out = None
        else:
            attention_mask_out = attention_mask_padded.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids_out = None
        else:
            position_ids_out = position_ids_padded

        return input_embeds_tensor, labels_out, attention_mask_out, position_ids_out

    def _pad_placeholder_masks(
        self,
        placeholder_masks: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        max_len = attention_mask.shape[1]
        padded_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        padding_side = getattr(self.config, "tokenizer_padding_side", "right")
        for idx, mask in enumerate(placeholder_masks):
            cur_len = int(mask.shape[0])
            if cur_len == 0:
                continue
            cur_mask = mask.to(device=padded_mask.device)
            if padding_side == "right":
                padded_mask[idx, :cur_len] = cur_mask
            else:
                padded_mask[idx, max_len - cur_len :] = cur_mask
        return padded_mask


AutoModelForCausalLM.register(UniMVUUnifiedConfig, UniMVUUnifiedForCausalLM)
