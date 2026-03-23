# This script holds all the model configs

import os

from collections import defaultdict
from dataclasses import dataclass, field, fields
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, List, Tuple
import warnings

import torch
import transformers

from transformers.trainer import logger

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass(frozen=True)
class DatasetConfig:
    """Static metadata describing modal coverage and fast-feature layout for a dataset."""

    name: str
    modalities: Tuple[str, ...] = ("video",)
    fast_feat_sequence: Tuple[str, ...] = ("video_vae",)
    second_sides_sequence: Tuple[Optional[str], ...] = tuple()
    video_backbone_sequence: Tuple[Optional[str], ...] = tuple()

    def fast_feat_for_index(self, occurrence: int) -> Optional[str]:
        if not self.fast_feat_sequence:
            return None
        if occurrence < len(self.fast_feat_sequence):
            return self.fast_feat_sequence[occurrence]
        return self.fast_feat_sequence[-1]

    def second_side_for_index(self, occurrence: int) -> Optional[str]:
        if not self.second_sides_sequence:
            return None
        if occurrence < len(self.second_sides_sequence):
            return self.second_sides_sequence[occurrence]
        return self.second_sides_sequence[-1]


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "avsd": DatasetConfig(
        name="avsd",
        modalities=("video", "audio"),
        fast_feat_sequence=("audio",),
        second_sides_sequence=("audio",),
        video_backbone_sequence=("safe_video_reader",),
    ),
    "avqa": DatasetConfig(
        name="avqa",
        modalities=("video", "audio"),
        fast_feat_sequence=("audio",),
        second_sides_sequence=("audio",),
        video_backbone_sequence=("tv",),
    ),
    "music_avqa": DatasetConfig(
        name="music_avqa",
        modalities=("video", "audio"),
        fast_feat_sequence=("audio",),
        second_sides_sequence=("audio",),
        video_backbone_sequence=("safe_video_reader",),
    ),
    "scanqa": DatasetConfig(
        name="scanqa",
        modalities=("video", "3d_feature"),
        fast_feat_sequence=("3d_feature",),
        video_backbone_sequence=("safe_video_reader",),
    ),
    "sqa3d": DatasetConfig(
        name="sqa3d",
        modalities=("video", "3d_feature"),
        fast_feat_sequence=("3d_feature",),
        video_backbone_sequence=("safe_video_reader",),
    ),
    "llava_video": DatasetConfig(
        name="llava_video",
        modalities=("video", "dense_video"),
        fast_feat_sequence=("dense_video",),
        video_backbone_sequence=("safe_video_reader",),
    ),
}


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



@dataclass
class DataArguments:
    '''
        For the LLaVA dataset
    '''
    
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    

@dataclass
class VideoDataArguments:
    '''
        For the normal video instruction tuning dataset
    '''    
    
    annotation_path: str = field(default=None, metadata={"help": "Path to the annotation."})
    feat_mapping_path: str = field(default=None, metadata={"help": "Path to the mapping."})
    data_root: Optional[str] = field(default=None)
    use_video_feat: bool = False
    lazy_preprocess: bool = False
    is_multimodal: bool = True

    image_size = (224, 224)
    transform_name = 'resize_crop'
    num_frames = 50 # was the 17 * 10
    # num_frames=32
    # frame_interval=1


@dataclass
class VideoFeatDataArguments:
    '''
        For the video instruction tuning dataset with pre-extracted feature.
        Probably we will also use the video data, if we use the llava-onevision.
    ''' 
    ### if and only if we define like this
    # use_fast_feat: bool = True, the variable could be recognized by the hugging face
    
    annotation_path: str = field(default=None, metadata={"help": "Path to the annotation."})
    fast_path_mapping_path: str = field(default=None, metadata={"help": "Path to the fast path data mapping file."})
    slow_path_mapping_path: str = field(default=None, metadata={"help": "Path to the video mapping."})
    data_root: Optional[str] = field(default=None, metadata={"help": "Path to the video feature."})
    slow_path_data_root: Optional[str] = field(default=None, metadata={"help": "Path to the slowpath data."})
    data_sample_ratio: Optional[str] = field(default=None, metadata={"help": "ratio of each dataset sampled"})
    video_loading_backbone: str = field(default='decord')
    # defined some hyper of the dataset
    use_fast: bool = False      # a special version which directly send the video frame in
    use_fast_feat: bool = True  # use pre-extracted video feature in the training
    use_slow: bool = False      # use the image and the image backbone for the training
    use_slow_feat: bool = False # use pre-extracted video feature in the training
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    prepare_qid: bool = True

    # For fast loading
    fast_feat_type: str = 'video_vae'
    original_feat_fps: int = 24                 # for video-vae feature
    training_feat_fps: int = 4
    min_fast_frame_num: int = 32
    exclude_languagebind_cls_token: bool = True # for languagebind feature: this is for excluding the first token from the languagebind during the training
    
    # For slow loading
    frames_upbound: int = 32
    force_sample: int = True # setting frames_upbound and force_sample will force the number of frame to be 32
    video_fps: int = 1       # like a scaling factor 
    
    image_size = (224, 224) # TODO: be careful about this part check meaningless config
    transform_name = 'resize_crop'  # TODO: be careful about this part check meaningless config
    # num_frames = 50 # was the 17 * 10

    ### add for second side channels
    use_second_sides: bool = False
    second_sides_type: str = 'audio'
    second_sides_data_root: Optional[str] = field(default=None, metadata={"help": "Path to the second sides data."})
    modalities: Optional[List[str]] = field(default=None, metadata={"help": "Modalities provided by the dataset."})


@dataclass
class MixedVideoFeatDataArguments(VideoFeatDataArguments):
    """Mixed-dataset variant of :class:`VideoFeatDataArguments` with broadcast helpers."""

    annotation_path: str = field(default=None)
    fast_path_mapping_path: str = field(default=None)
    slow_path_mapping_path: Optional[str] = field(default=None)
    data_root: Optional[str] = field(default=None)
    slow_path_data_root: Optional[str] = field(default=None)
    data_sample_ratio: Optional[str] = field(default=None)
    second_sides_data_root: Optional[str] = field(default=None)
    datasets: List[str] = field(
        default_factory=list,
        metadata={"help": "Dataset identifiers aligned with annotation_path order. Use 'name*COUNT' to repeat without redundancy."},
    )
    second_sides_type: Optional[str] = field(default="audio")
    dataset_fast_feat_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override dataset->fast modality mapping, e.g. 'avsd:audio,scanqa:3d_feature' or 'llava_video:languagebind|audio'."
        },
    )
    dataset_second_sides_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override dataset->second side channel mapping, support multiple via '|', e.g. 'avsd:audio|audio'."
        },
    )
    dataset_video_backbone_map: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override dataset->video loading backbone, e.g. 'avqa:tv,llava_video:decord'. "
                "Multiple occurrences for repeated datasets can be provided via '|', e.g. 'dataset:tv|decord'."
            )
        },
    )

    _DEFAULT_FAST_FEAT_MAP = {
        name: list(cfg.fast_feat_sequence)
        for name, cfg in DATASET_CONFIGS.items()
        if cfg.fast_feat_sequence
    }
    _DEFAULT_SECOND_SIDES_MAP = {
        name: [side for side in cfg.second_sides_sequence]
        for name, cfg in DATASET_CONFIGS.items()
        if cfg.second_sides_sequence
    }
    _DEFAULT_VIDEO_BACKBONE_MAP = {
        name: [backbone for backbone in cfg.video_backbone_sequence if backbone]
        for name, cfg in DATASET_CONFIGS.items()
        if cfg.video_backbone_sequence
    }
    _BROADCAST_FIELDS = {
        "annotation_path",
        "fast_path_mapping_path",
        "slow_path_mapping_path",
        "data_root",
        "slow_path_data_root",
        "data_sample_ratio",
        "second_sides_data_root",
        "second_sides_type",
        "modalities",
    }

    def _num_datasets(self) -> int:
        value = self.annotation_path
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("annotation_path list must not be empty.")
            return len(value)
        if value is None:
            raise ValueError("annotation_path must be provided.")
        return 1

    def _broadcast_field(self, name: str, total: int) -> List[Optional[str]]:
        value = getattr(self, name)
        if value is None:
            return [None] * total
        if isinstance(value, (list, tuple)):
            if len(value) == total:
                return list(value)
            if len(value) == 1:
                return [value[0]] * total
            raise ValueError(f"Field '{name}' expects {total} values, got {len(value)}.")
        return [value] * total

    def _shared_kwargs(self) -> Dict[str, object]:
        shared = {}
        for f in fields(VideoFeatDataArguments):
            if f.name not in self._BROADCAST_FIELDS:
                shared[f.name] = getattr(self, f.name)
        return shared

    def to_per_dataset_arguments(self) -> List[VideoFeatDataArguments]:
        total = self._num_datasets()
        shared = self._shared_kwargs()
        broadcast = {
            name: self._broadcast_field(name, total) for name in self._BROADCAST_FIELDS
        }
        fast_lookup = self._build_fast_feat_lookup()
        second_lookup = self._build_second_sides_lookup()
        video_backbone_lookup = self._build_video_backbone_lookup()
        dataset_names = self._resolve_dataset_names(total)
        extra_attrs = self._shared_extra_attrs()

        dataset_args: List[VideoFeatDataArguments] = []
        dataset_occurrence: Dict[str, int] = defaultdict(int)
        for idx in range(total):
            kwargs = dict(shared)
            for name, values in broadcast.items():
                kwargs[name] = values[idx]
            dataset_name = dataset_names[idx]
            occurrence_index = dataset_occurrence[dataset_name] if dataset_name else 0
            if dataset_name:
                dataset_occurrence[dataset_name] += 1

            kwargs["fast_feat_type"] = self._infer_fast_feat_type(
                kwargs.get("annotation_path"), dataset_name, fast_lookup, occurrence_index
            )
            kwargs["second_sides_type"] = self._infer_second_sides_type(
                kwargs.get("annotation_path"), dataset_name, second_lookup, occurrence_index
            )
            kwargs["video_loading_backbone"] = self._infer_video_loading_backbone(
                kwargs.get("annotation_path"), dataset_name, video_backbone_lookup, occurrence_index
            )
            inferred_modalities = self._infer_modalities(dataset_name)
            if inferred_modalities is not None and not kwargs.get("modalities"):
                kwargs["modalities"] = inferred_modalities
            per_arg = VideoFeatDataArguments(**kwargs)
            for attr, value in extra_attrs.items():
                setattr(per_arg, attr, value)
            dataset_args.append(per_arg)
        return dataset_args

    def _infer_fast_feat_type(
        self,
        annotation_path: Optional[str],
        dataset_name: Optional[str],
        fast_lookup: Dict[str, List[Optional[str]]],
        occurrence_index: int,
    ) -> str:
        if dataset_name:
            name_key = dataset_name.lower()
            if name_key in fast_lookup:
                candidates = [feat for feat in fast_lookup[name_key] if feat is not None]
                if candidates:
                    if occurrence_index < len(candidates):
                        return candidates[occurrence_index]
                    return candidates[-1]
        if not annotation_path:
            warnings.warn("No annotation path and dataset name provided, using default fast feat type.")
            return self.fast_feat_type
        marker = Path(annotation_path).stem.lower()
        for key, feat in fast_lookup.items():
            if key in marker:
                seq = [val for val in feat if val is not None]
                if seq:
                    warnings.warn(f"Found dataset name {key} in annotation path, using {seq[0]} as fast feat type.")
                    return seq[0]
        parent_tokens = [part.lower() for part in Path(annotation_path).parts]
        for key, feat in fast_lookup.items():
            if any(key in token for token in parent_tokens):
                seq = [val for val in feat if val is not None]
                if seq:
                    warnings.warn(f"Found dataset name {key} in annotation path, using {seq[0]} as fast feat type.")
                    return seq[0]
        warnings.warn("No dataset name or annotation path token in fast feat lookup, using default fast feat type.")
        return self.fast_feat_type

    def _infer_second_sides_type(
        self,
        annotation_path: Optional[str],
        dataset_name: Optional[str],
        second_lookup: Dict[str, List[Optional[str]]],
        occurrence_index: int,
    ) -> str:
        if dataset_name:
            name_key = dataset_name.lower()
            if name_key in second_lookup:
                candidates = second_lookup[name_key]
                if candidates:
                    if occurrence_index < len(candidates):
                        return candidates[occurrence_index] or ""
                    return candidates[-1] or ""
        if not annotation_path:
            return self.second_sides_type or ""
        marker = Path(annotation_path).stem.lower()
        for key, side in second_lookup.items():
            if key in marker and side:
                return side[0] or ""
        parent_tokens = [part.lower() for part in Path(annotation_path).parts]
        for key, side in second_lookup.items():
            if any(key in token for token in parent_tokens) and side:
                return side[0] or ""
        return self.second_sides_type or ""

    def _infer_video_loading_backbone(
        self,
        annotation_path: Optional[str],
        dataset_name: Optional[str],
        backbone_lookup: Dict[str, List[str]],
        occurrence_index: int,
    ) -> str:
        if dataset_name:
            name_key = dataset_name.lower()
            if name_key in backbone_lookup:
                candidates = backbone_lookup[name_key]
                if candidates:
                    if occurrence_index < len(candidates):
                        return candidates[occurrence_index]
                    return candidates[-1]
        if not annotation_path:
            return self.video_loading_backbone
        marker = Path(annotation_path).stem.lower()
        for key, backs in backbone_lookup.items():
            if key in marker and backs:
                return backs[0]
        parent_tokens = [part.lower() for part in Path(annotation_path).parts]
        for key, backs in backbone_lookup.items():
            if any(key in token for token in parent_tokens) and backs:
                return backs[0]
        return self.video_loading_backbone

    def _build_fast_feat_lookup(self) -> Dict[str, List[Optional[str]]]:
        lookup = dict(self._DEFAULT_FAST_FEAT_MAP)
        if not self.dataset_fast_feat_map:
            return lookup
        for entry in (chunk.strip() for chunk in self.dataset_fast_feat_map.split(",")):
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(
                    f"Invalid dataset_fast_feat_map entry '{entry}'. Expected 'name:feat'."
                )
            name, feat_spec = entry.split(":", maxsplit=1)
            name = name.strip().lower()
            if not name:
                raise ValueError(
                    f"Invalid dataset_fast_feat_map entry '{entry}'. Empty name or feat sequence."
                )
            feats = [token.strip() for token in feat_spec.split("|") if token.strip()]
            if not feats:
                raise ValueError(
                    f"Invalid dataset_fast_feat_map entry '{entry}'. Empty modality list."
                )
            lookup[name] = feats
        return lookup

    def _build_second_sides_lookup(self) -> Dict[str, List[Optional[str]]]:
        lookup = dict(self._DEFAULT_SECOND_SIDES_MAP)
        if not self.dataset_second_sides_map:
            return lookup
        for entry in (chunk.strip() for chunk in self.dataset_second_sides_map.split(",")):
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(
                    f"Invalid dataset_second_sides_map entry '{entry}'. Expected 'name:feat'."
                )
            name, feat_spec = entry.split(":", maxsplit=1)
            name = name.strip().lower()
            if not name:
                raise ValueError(
                    f"Invalid dataset_second_sides_map entry '{entry}'. Empty name or feat sequence."
                )
            feats = [token.strip() if token.strip() else None for token in feat_spec.split("|")]
            if not feats:
                raise ValueError(
                    f"Invalid dataset_second_sides_map entry '{entry}'. Empty modality list."
                )
            lookup[name] = feats
        return lookup

    def _build_video_backbone_lookup(self) -> Dict[str, List[str]]:
        lookup = dict(self._DEFAULT_VIDEO_BACKBONE_MAP)
        if not self.dataset_video_backbone_map:
            return lookup
        for entry in (chunk.strip() for chunk in self.dataset_video_backbone_map.split(",")):
            if not entry:
                continue
            if ":" not in entry:
                raise ValueError(
                    f"Invalid dataset_video_backbone_map entry '{entry}'. Expected 'name:backbone'."
                )
            name, backbone_spec = entry.split(":", maxsplit=1)
            name = name.strip().lower()
            if not name:
                raise ValueError(
                    f"Invalid dataset_video_backbone_map entry '{entry}'. Empty dataset name."
                )
            backbones = [token.strip() for token in backbone_spec.split("|") if token.strip()]
            if not backbones:
                raise ValueError(
                    f"Invalid dataset_video_backbone_map entry '{entry}'. Empty backbone list."
                )
            lookup[name] = backbones
        return lookup

    def _infer_modalities(self, dataset_name: Optional[str]) -> Optional[List[str]]:
        if not dataset_name:
            return None
        config = DATASET_CONFIGS.get(dataset_name.lower())
        if not config:
            return None
        return list(config.modalities)

    def _resolve_dataset_names(self, total: int) -> List[Optional[str]]:
        if not self.datasets:
            return [None] * total

        expanded: List[Optional[str]] = []
        for raw_token in self.datasets:
            if raw_token is None:
                expanded.append(None)
                continue
            token = raw_token.strip()
            if not token:
                expanded.append(None)
                continue
            if "*" in token:
                name, count_str = token.split("*", maxsplit=1)
                name = name.strip()
                count_str = count_str.strip()
                if not count_str:
                    raise ValueError(f"Invalid dataset token '{raw_token}'. Expected '*count' after name.")
                try:
                    repeat = int(count_str)
                except ValueError as exc:
                    raise ValueError(f"Invalid repeat count in dataset token '{raw_token}'.") from exc
                if repeat <= 0:
                    raise ValueError(f"Repeat count must be positive in dataset token '{raw_token}'.")
                expanded.extend([name if name else None] * repeat)
            else:
                expanded.append(token)

        if len(expanded) == 1 and total > 1:
            expanded *= total

        if len(expanded) != total:
            raise ValueError(
                f"Expanded dataset identifiers ({len(expanded)}) do not match annotation entries ({total}). "
                "Use 'name*COUNT' to repeat identifiers without redundancy."
            )
        return [name.lower() if name else None for name in expanded]

    def _shared_extra_attrs(self) -> Dict[str, object]:
        field_names = {f.name for f in fields(MixedVideoFeatDataArguments)}
        field_names.update({f.name for f in fields(VideoFeatDataArguments)})
        extras: Dict[str, object] = {}
        for attr_name, attr_value in self.__dict__.items():
            if (
                attr_name.startswith("_")
                or attr_name in field_names
                or callable(attr_value)
                or attr_value is None
            ):
                continue
            extras[attr_name] = attr_value
        return extras


@dataclass
class VideoTrainingArguments(transformers.TrainingArguments):
    '''
        For the normal video instruction tuning training argument
    '''        
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False) # Alert: this should not be removed
    # freeze_mm_mlp_adapter: bool = field(default=False)
    # mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    temporal_aggregator_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle mixed datasets when constructing the training dataloader. Disable to use simple concatenation."},
    )
    max_grad_norm: float = 0.1
    mix_sampling_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Exponent for alpha-based sampling over mixed datasets. 1.0 keeps dataset-proportional sampling, 0.0 equalizes dataset contributions."
        },
    )
    full_determinism: bool = True
    seed: int = 42 
    dpo_alpha: float = field(default=1.0)
    beta: float = field(default=0.1)
    gamma: float = field(default=0.0)

    # Extra control over modules to train/save (non-LoRA)
    extra_trainable_modules: Optional[List[str]] = field(default=None, metadata={"help": "List of substrings of module names to set requires_grad=True for training."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    '''
        For the LLaVA training argument
    ''' 
    
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


@dataclass
class VideoFeatModelArgumentsUniMVU:
    '''
        For the UniMVU VLMM Qwen2 training argument
    '''
    model_type: str = field(default="unimvu")
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    video_tower: Optional[str] = field(default=None)                 # the type of the vision backbone, if None, it will use the vision tower
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    ######################### feature merging config ############################################
    input_dim: Optional[int] = field(default=1024)
    feat_combine_method: str = field(default="concat")
    
    ######################### modality_aggregator config ############################################
    num_cross_modality_hidden_layers: Optional[int] = field(default=1)
    support_modalities: Optional[List[str]] = field(default_factory=list)

    # Individual modality_aggregator_config parameters (can be overridden via command line)
    modality_aggregator_hidden_size: Optional[int] = field(default=896)
    modality_aggregator_num_heads: Optional[int] = field(default=14)
    modality_aggregator_num_key_value_heads: Optional[int] = field(default=14)
    modality_aggregator_rope_theta: Optional[int] = field(default=250000)
    modality_aggregator_attention_dropout: Optional[float] = field(default=0.0)
    modality_aggregator_modality_token_num: Optional[int] = field(default=1)
    modality_input_dims: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Per-modality projector input dimensions. "
                "Accepts comma-separated pairs like 'video:1024,audio:768,3d_feature:1024'. "
                "Unspecified modalities fall back to `input_dim`."
            )
        },
    )
    def __post_init__(self):
        # Build modality_aggregator_config dict from individual parameters
        self.modality_aggregator_config = dict(
            hidden_size=self.modality_aggregator_hidden_size,
            num_heads=self.modality_aggregator_num_heads,
            num_key_value_heads=self.modality_aggregator_num_key_value_heads,
            rope_theta=self.modality_aggregator_rope_theta,
            attention_dropout=self.modality_aggregator_attention_dropout,
            modality_token_num=self.modality_aggregator_modality_token_num,
        )

@dataclass
class VideoFeatModelArgumentsUniMVU_Uni:
    '''
        For the UniMVU VLMM Qwen2 training argument
    '''
    model_type: str = field(default="unimvu_uni")
    modality_input_dims: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Per-modality projector input dimensions. "
                "Accepts comma-separated pairs like 'video:1024,audio:768,3d_feature:1024'. "
                "Unspecified modalities fall back to `input_dim`."
            )
        },
    )

@dataclass
class VideoFeatModelArgumentsUniMVU_7B:
    '''
        For the UniMVU VLMM Qwen2 training argument
    '''
    model_type: str = field(default="unimvu_uni")
    ############################ Addition tokens config #############################################
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    tune_addition_token_embeddings: bool = field(default=False)
    ############################ LLM config #########################################################
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") # the path of the LLM
    # freeze_backbone: bool = field(default=False)                         # freeze the LLM
    version: Optional[str] = field(default="v0")                           # the tokenizer version
    ############################ Video Tower Config #################################################
    # video_tower: Optional[str] = field(default='opensora')                 # the type of the vision backbone
    video_tower: Optional[str] = field(default=None)                 # the type of the vision backbone, if None, it will use the vision tower
    vae = None
    text_encoder = None
    diffu = None
    diffu_t = 0 # meaningless config
    diffu_extract_depth = 0 # meaningless config
 
    ############################ Vision part config ###########################################
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    mm_projector_type: Optional[str] = field(default="mlp2x_gelu")
    mm_patch_merge_type: Optional[str] = field(default="spatial_unpad")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # pretrain_vision_modules: Optional[str] = field(default=None) ## load the vision backbone, the MLP, and the newline

    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None) 
    mm_newline_position: Optional[str] = field(default="grid")  

    ######################### feature merging config ############################################
    input_dim: Optional[int] = field(default=1024)
    feat_combine_method: str = field(default="concat")
    
    ######################### modality_aggregator config ############################################
    num_cross_modality_hidden_layers: Optional[int] = field(default=1)
    support_modalities: Optional[List[str]] = field(default_factory=list)
    
    modality_aggregator_hidden_size: Optional[int] = field(default=3584)
    modality_aggregator_num_heads: Optional[int] = field(default=14)
    modality_aggregator_num_key_value_heads: Optional[int] = field(default=14)
    modality_aggregator_rope_theta: Optional[int] = field(default=250000)
    modality_aggregator_attention_dropout: Optional[float] = field(default=0.0)
    modality_aggregator_modality_token_num: Optional[int] = field(default=8)
    
    def __post_init__(self):
        # Build modality_aggregator_config dict from individual parameters
        self.modality_aggregator_config = dict(
            hidden_size=self.modality_aggregator_hidden_size,
            num_heads=self.modality_aggregator_num_heads,
            num_key_value_heads=self.modality_aggregator_num_key_value_heads,
            rope_theta=self.modality_aggregator_rope_theta,
            attention_dropout=self.modality_aggregator_attention_dropout,
            modality_token_num=self.modality_aggregator_modality_token_num,
        )

@dataclass
class VideoFeatModelArgumentsUniMVU_Uni_7B(VideoFeatModelArgumentsUniMVU_7B):
    """
    Extends UniMVU Uni arguments with 7B model.
    """

    model_type: str = field(default="unimvu_uni")
    modality_input_dims: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Per-modality projector input dimensions. "
                "Accepts comma-separated pairs like 'video:1024,audio:768,3d_feature:1024'. "
                "Unspecified modalities fall back to `input_dim`."
            )
        },
    )


# the mapping between the name of the class and the class function
MODEL_ARGUMENTS_MAPPING = {
    # UniMVU (Unified Multimodal Video Understanding) configurations
    'VideoFeatModelArgumentsUniMVU': VideoFeatModelArgumentsUniMVU,
    'VideoFeatModelArgumentsUniMVU_Uni': VideoFeatModelArgumentsUniMVU_Uni,
    'VideoFeatModelArgumentsUniMVU_7B': VideoFeatModelArgumentsUniMVU_7B,
    'VideoFeatModelArgumentsUniMVU_Uni_7B': VideoFeatModelArgumentsUniMVU_Uni_7B,

}

DATA_ARGUMENTS_MAPPING = {
    'default': VideoFeatDataArguments,
    'VideoFeatDataArguments': VideoFeatDataArguments,
    'VideoFeatMixedDataArguments': MixedVideoFeatDataArguments,
}

TRAINING_ARGUMENTS_MAPPING = {
    'default': VideoTrainingArguments,
    'VideoTrainingArguments': VideoTrainingArguments
}

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3_with_state_dict(state_dict, special_key=[]):
    # this is a special version of the get_peft_state_non_lora_maybe_zero_3
    # it use the state dict to save the model,
    # this will save the 'running_mean', 'running_var', 'num_batches_tracked' in batch norm
    # special_key = ['running_mean', 'running_var', 'num_batches_tracked']
    non_lora = {k: state_dict[k] for k in state_dict if "lora_" not in k}
    # filter the state dict with the key
    to_return = {}
    for k, t in non_lora.items():
        if any(key_match in k for key_match in special_key):
            to_return[k] = t

    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3_with_state_dict(state_dict, keys_to_match):
    # this is a special version of saving the params with given key
    # it will use the state_dict to save the parameters
    # this will also save the 'running_mean', 'running_var', 'num_batches_tracked' in the batch norm
    to_return = {k: state_dict[k] for k in state_dict if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_temporal_aggregator", False) or getattr(trainer.args, "extra_trainable_modules", False):
        # Only save Adapter
        keys_to_match = ['temporal_aggregator', 
                         'self_attn.v_kv_proj', # for the mplug-owl3
                         'self_attn.gate_proj',
                         'input_mapping'] + (getattr(trainer.args, 'extra_trainable_modules', None) or [])

        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def safe_save_model_for_hf_videotrainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_temporal_aggregator", False) or getattr(trainer.args, "extra_trainable_modules", False):
        # Only save Adapter
        keys_to_match = ['temporal_aggregator', 
                        'self_attn.v_kv_proj', # for the mplug-owl3
                        'self_attn.gate_proj',
                        'input_mapping'] + (getattr(trainer.args, 'extra_trainable_modules', None) or [])
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3_with_state_dict(trainer.model.state_dict(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def parse_argument_classes(sys_args, return_name=False):
    # This function aims to takes all sys arguments input,
    # figure out the model_class, data_class, training_class 
    # and retunr the remaining arguments
    
    # parse the arugment for the model
    remaining_args = []
    model_class_name = 'default'
    data_class_name = 'default'
    training_class_name = 'default'
    i = 0
    while i < len(sys_args):
        ele = sys_args[i]
        if ele == '--model_class':
            assert i+1 < len(sys_args) # assert is not empty
            model_class_name = sys_args[i+1]
            i += 2
        elif ele == '--data_class':
            assert i+1 < len(sys_args) # assert is not empty
            data_class_name = sys_args[i+1]
            i += 2
        elif ele == '--training_class':
            assert i+1 < len(sys_args) # assert is not empty
            training_class_name = sys_args[i+1]
            i += 2
        else:
            remaining_args.append(ele)
            i += 1
    
    # find the class through mapping
    model_arg_class = MODEL_ARGUMENTS_MAPPING[model_class_name]
    data_arg_class = DATA_ARGUMENTS_MAPPING[data_class_name]
    training_arg_class = TRAINING_ARGUMENTS_MAPPING[training_class_name]
    logger.info('model_class_name: ' + model_class_name + ' data_class_name: ' + data_class_name + ' training_class_name: ' + training_class_name)
    
    ##################################################### parse the dataset information #################################################################
    # find all the argument head, and mark the position of the --fast_path_mapping_path and --data_root
    all_argument_head_pos = []
    annotation_path_pos = None            # the position in the remaining_args
    annotation_path_pos_in_list = None    # the position in the all_argument_head_pos
    fast_path_mapping_path_pos = None          # the position in the remaining_args
    fast_path_mapping_path_pos_in_list = None  # the position in the all_argument_head_pos
    slow_path_mapping_path_pos = None         # the position in the remaining_args
    slow_path_mapping_path_pos_in_list = None # the position in the all_argument_head_pos
    data_root_pos = None                  # the position in the remaining_args
    data_root_pos_in_list = None          # the position in the all_argument_head_pos
    slow_path_data_root_pos = None            # the position in the remaining_args
    slow_path_data_root_pos_in_list = None    # the position in the all_argument_head_pos
    
    
    for i, curr_arg in enumerate(remaining_args):
        if curr_arg.startswith('--'):
            all_argument_head_pos.append(i)
        if curr_arg == '--annotation_path':
            annotation_path_pos_in_list = len(all_argument_head_pos) - 1
            annotation_path_pos = i
        if curr_arg == '--fast_path_mapping_path':
            fast_path_mapping_path_pos_in_list = len(all_argument_head_pos) - 1
            fast_path_mapping_path_pos = i
        if curr_arg == '--slow_path_mapping_path':
            slow_path_mapping_path_pos_in_list = len(all_argument_head_pos) - 1
            slow_path_mapping_path_pos = i            
        if curr_arg == '--data_root':
            data_root_pos_in_list = len(all_argument_head_pos) - 1
            data_root_pos = i
        if curr_arg == '--slow_path_data_root':
            slow_path_data_root_pos_in_list = len(all_argument_head_pos) - 1
            slow_path_data_root_pos = i            

    # figure out the len of the arugment input for the data_root_pos and all_argument_head_pos
    assert annotation_path_pos is not None
    assert annotation_path_pos_in_list is not None
    assert fast_path_mapping_path_pos is not None 
    assert fast_path_mapping_path_pos_in_list is not None
    assert data_root_pos is not None
    assert data_root_pos_in_list is not None
    # p.s  slow_path_mapping_path_pos, slow_path_mapping_path_pos_in_list, slow_path_data_root_pos, slow_path_data_root_pos_in_list could be None
    
    annotation_path_start = annotation_path_pos + 1
    annotation_path_end = all_argument_head_pos[annotation_path_pos_in_list + 1] \
        if annotation_path_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)    # if it is the last arguments
    
    fast_path_mapping_path_start = fast_path_mapping_path_pos + 1
    fast_path_mapping_path_end = all_argument_head_pos[fast_path_mapping_path_pos_in_list + 1] \
        if fast_path_mapping_path_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)  # if it is the last arguments
    
    data_root_start = data_root_pos + 1
    data_root_end = all_argument_head_pos[data_root_pos_in_list + 1] \
        if data_root_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)          # if it is the last arguments
    
    num_of_annotation_path = annotation_path_end - annotation_path_start
    num_of_fast_path_mapping_path = fast_path_mapping_path_end - fast_path_mapping_path_start
    num_of_data_root = data_root_end - data_root_start
    
    # assert the len of the --data_root == --fast_path_mapping_path
    # ipdb.set_trace() # check the argument parser
    assert num_of_fast_path_mapping_path == num_of_data_root == num_of_annotation_path
    assert num_of_fast_path_mapping_path > 0
    
    if slow_path_mapping_path_pos is not None: # the input has the video mapping
        slow_path_mapping_path_start = slow_path_mapping_path_pos + 1
        slow_path_mapping_path_end = all_argument_head_pos[slow_path_mapping_path_pos_in_list + 1] \
            if slow_path_mapping_path_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)  # if it is the last arguments        
    
        assert slow_path_data_root_pos is not None
        slow_path_data_root_start = slow_path_data_root_pos + 1
        slow_path_data_root_end = all_argument_head_pos[slow_path_data_root_pos_in_list + 1] \
            if slow_path_data_root_pos_in_list + 1 < len(all_argument_head_pos) else len(remaining_args)     # if it is the last arguments
        
        num_of_slow_path_mapping_path = slow_path_mapping_path_end - slow_path_mapping_path_start
        num_of_slow_path_data_root = slow_path_data_root_end - slow_path_data_root_start
        
        assert num_of_annotation_path ==  num_of_slow_path_mapping_path == num_of_slow_path_data_root
    else:
        num_of_slow_path_mapping_path = 0
        num_of_slow_path_data_root = 0
    
    # assign the result to the data_arg_class
    if num_of_fast_path_mapping_path > 1:
        annotation_path_set = remaining_args[annotation_path_start:annotation_path_end]
        fast_path_mapping_path_set = remaining_args[fast_path_mapping_path_start: fast_path_mapping_path_end]
        data_root_set = remaining_args[data_root_start: data_root_end]
        filter_set = annotation_path_set + fast_path_mapping_path_set + data_root_set + ['--annotation_path', '--fast_path_mapping_path', '--data_root']
        
        if num_of_slow_path_mapping_path > 0:
            slow_path_mapping_path_set = remaining_args[slow_path_mapping_path_start: slow_path_mapping_path_end]
            slow_path_data_root_set = remaining_args[slow_path_data_root_start: slow_path_data_root_end]
            filter_set = filter_set + slow_path_mapping_path_set + slow_path_data_root_set + ['--slow_path_mapping_path', '--slow_path_data_root']
            
    else:
        annotation_path_set = remaining_args[annotation_path_start]
        fast_path_mapping_path_set = remaining_args[fast_path_mapping_path_start]
        data_root_set = remaining_args[data_root_start]
        filter_set = ['--annotation_path', '--fast_path_mapping_path', '--data_root', remaining_args[annotation_path_start], remaining_args[fast_path_mapping_path_start], remaining_args[data_root_start]]
        
        if num_of_slow_path_mapping_path > 0:
            slow_path_mapping_path_set = remaining_args[slow_path_mapping_path_start]
            slow_path_data_root_set = remaining_args[slow_path_data_root_start]
            filter_set = filter_set + [slow_path_mapping_path_set, slow_path_data_root_set, '--slow_path_mapping_path', '--slow_path_data_root']
        
    dataset_argument_filtered = [ele for ele in remaining_args if ele not in filter_set]
    
    ######################################################## parser the rest of the auguments ##############################################
    parser = transformers.HfArgumentParser(
        (model_arg_class, data_arg_class, training_arg_class))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=dataset_argument_filtered)    
    
    # assign the dataset value back to the class
    data_args.annotation_path = annotation_path_set
    data_args.fast_path_mapping_path = fast_path_mapping_path_set
    data_args.data_root = data_root_set
    if num_of_slow_path_mapping_path > 0:
        data_args.slow_path_mapping_path = slow_path_mapping_path_set
        data_args.slow_path_data_root = slow_path_data_root_set
    if return_name:
        return model_args, data_args, training_args, model_class_name, data_class_name, training_class_name
    else:
        return model_args, data_args, training_args
