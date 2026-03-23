# This script holds the structural implementation of the UniMVU.

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from libs.constants import IMAGE_TOKEN_INDEX
from libs.mm_utils import split_list_lengths
from einops import rearrange
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from libs.model.multimodal_base.multimodal_base_model import LMMMetaModel, LMMMetaForCausalLM


class UniMVUQwen2Config(Qwen2Config):
    model_type = "unimvu"
    
    def __init__(self, model_config=None, **kwargs):
        # Forward all base LLM fields so tensor shapes match checkpoint weights
        super(UniMVUQwen2Config, self).__init__(**kwargs)

        # Copy all attributes from model_config to self
        # This ensures that all command-line arguments are preserved in the config
        if model_config is not None:
            for key, value in model_config.__dict__.items():
                if not key.startswith('__') and not key.startswith('_'):
                    # Only set if not already set by kwargs or parent class
                    if not hasattr(self, key) or getattr(self, key) is None:
                        setattr(self, key, value)

        # Set UniMVU-specific fields with proper defaults
        self.num_cross_modality_hidden_layers = getattr(self, 'num_cross_modality_hidden_layers', 1)
        self.support_modalities = getattr(self, 'support_modalities', ["video", "audio"])
        self.input_dim = getattr(self, 'input_dim', 1024)
        
        # Set modality_aggregator_config with proper default
        if not hasattr(self, 'modality_aggregator_config') or self.modality_aggregator_config is None:
            self.modality_aggregator_config = {
                'hidden_size': 896,
                'num_heads': 14,
                'num_key_value_heads': 14,
                'rope_theta': 250000,
                'attention_dropout': 0.0,
                'modality_token_num': 1,
            }

class UniMVUQwen2Model(LMMMetaModel, Qwen2Model):
    config_class = UniMVUQwen2Config
    
    def __init__(self, config: Qwen2Config):
        super(UniMVUQwen2Model, self).__init__(config)

class UniMVUQwen2ForCausalLM(Qwen2ForCausalLM, LMMMetaForCausalLM):
    """
    UniMVUQwen2ForCausalLM is the model for UniMVU with Qwen2 for causal language modeling.
    Qwen2ForCausalLM handles the LLM part.
    LMMMetaForCausalLM handles the multimodal processing.
    """
    config_class = UniMVUQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = UniMVUQwen2Model(config)
        
        total_layers = getattr(config, "num_hidden_layers", None)
        if total_layers is None:
            raise ValueError("config.num_hidden_layers must be set for Qwen2")
        cross_n = int(getattr(config, "num_cross_modality_hidden_layers", 0) or 0)
        cross_n = max(0, min(cross_n, total_layers))

        # Initialize learnable modality special tokens using ParameterDict for elegant parameter management
        modalities = config.support_modalities.split(",") if isinstance(config.support_modalities, str) \
                     else config.support_modalities if isinstance(config.support_modalities, list) \
                     else ["video"]
        
        # Store modality names for later access
        self.supported_modalities = modalities
        self.modality_token_num = config.modality_aggregator_config['modality_token_num']

        self.modality_tokens = nn.Parameter(self._init_modality_token(config.hidden_size, self.modality_token_num), requires_grad=True)

        # Initialize the modality special token aggregator
        self.modality_special_token_aggregator = ModalityTokenAggregator(config.hidden_size, self.modality_token_num)

        # Create a single learnable aggregator shared across cross-modality layers.
        self.modality_aggregator = UniMVUModalityAggregator(**config.modality_aggregator_config)
        
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.input_mapping = nn.Linear(config.input_dim, config.hidden_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        
    def _init_modality_token(self, hidden_size: int, modality_token_num: int) -> torch.Tensor:
        """
        Initialize a learnable modality special token with proper scaling.
        
        Args:
            hidden_size: Dimension of the token embedding
            
        Returns:
            Initialized token tensor of shape [modality_token_num, hidden_size] with requires_grad=True
        """
        embed_std = 1.0 / math.sqrt(hidden_size)
        token = torch.randn(modality_token_num, hidden_size) * embed_std
        # Ensure requires_grad is set (nn.Parameter will maintain this)
        token.requires_grad_(True)
        return token
    
    def get_model(self):
        return self.model
    
    def get_modality_token(self, modality_name: str) -> nn.Parameter:
        """
        Get the learnable special token for a specific modality.
        
        Args:
            modality_name: Name of the modality (e.g., 'video', 'audio', '3d_feature')
            
        Returns:
            The modality-specific learnable token parameter [modality_token_num, hidden_size]
            
        Example:
            >>> video_token = model.get_modality_token('video')
            >>> audio_token = model.get_modality_token('audio')
        """
        if modality_name not in self.modality_tokens:
            raise KeyError(f"Modality '{modality_name}' not found. Available modalities: {list(self.modality_tokens.keys())}")
        return self.modality_tokens[modality_name]

    def _normalize_modalities_per_sample(self, modalities, batch_size: int) -> List[List[str]]:
        """
        Normalize the incoming `modalities` argument into a per-sample list of modality names.

        Accepts values such as:
        - None (treated as all samples having ['video'])
        - "video"
        - ["video", "audio"] (shared modalities for whole batch when batch_size==1)
        - ["video", "audio", ...] broadcast across batch
        - [["video", "audio"], ["video"]] (already per-sample)
        """

        def to_mod_list(entry) -> List[str]:
            if entry is None:
                return []
            if isinstance(entry, str):
                return [entry]
            if isinstance(entry, (list, tuple, set)):
                result = []
                for item in entry:
                    if item is None:
                        continue
                    if isinstance(item, str):
                        result.append(item)
                    else:
                        result.append(str(item))
                return result
            return [str(entry)]

        if modalities is None:
            normalized = [["video"] for _ in range(batch_size)]
        elif isinstance(modalities, str):
            normalized = [[modalities] for _ in range(batch_size)]
        elif isinstance(modalities, (list, tuple, set)):
            if batch_size == 1 and not any(isinstance(m, (list, tuple, set)) for m in modalities):
                normalized = [to_mod_list(modalities)]
            elif len(modalities) == batch_size:
                normalized = [to_mod_list(m) for m in modalities]
            elif len(modalities) == 1:
                base = to_mod_list(next(iter(modalities)))
                normalized = [list(base) for _ in range(batch_size)]
            else:
                collapsed = to_mod_list(modalities)
                normalized = [list(collapsed) for _ in range(batch_size)]
        else:
            normalized = [["video"] for _ in range(batch_size)]

        ensured = []
        for mods in normalized:
            ensured.append(mods if mods else ["video"])
        return ensured

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        video_feats: Optional[torch.FloatTensor] = None,
        video_feat_fps: Optional[torch.FloatTensor] = None,
        feat_frame_nums: Optional[torch.FloatTensor] = None,
        
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = ["video"],
        
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        video_metas = None,
        
        question_ids = None,
        question_lens = None,
        dpo_forward = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import ipdb
        # ipdb.set_trace() # check the bug
        if inputs_embeds is None: # this is the training or the first forward at inference
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                modality_indices,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                video_feats,
                video_feat_fps=video_feat_fps,
                feat_frame_nums=feat_frame_nums,
                question_ids=question_ids,
                question_lens=question_lens,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
                video_metas=video_metas,
            )
            # Note: modality aggregation now happens inside prepare_inputs_labels_for_multimodal

        # Propagate modality indices to layers for optional use in attention/masking
        current_modality_indices = locals().get('modality_indices', None)
        for i in range(self.config.num_cross_modality_hidden_layers):
            setattr(self.model.layers[i], 'modality_indices', current_modality_indices)

        if not dpo_forward:
            loss = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            
            return loss
        else: # dpo forward
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict            
                
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels            

    def _build_modality_indices(self,
                                input_ids: torch.Tensor,
                                attention_mask: Optional[torch.Tensor],
                                feature_token_types: list) -> torch.Tensor:
        """
        Construct per-token modality indices aligned with the final input embeddings.

        Modality encoding:
        - 0: text tokens
        - 1: video feature tokens
        - 2: other modality feature tokens
        Padded positions and prompt tokens are set to -1.

        Args:
            input_ids: [B, L] original token ids (before assembly)
            attention_mask: [B, L] boolean or 0/1 mask; if None, all positions valid
            feature_token_types: list of 1D tensors per-sample describing feature span types
                                  produced during feature combination (1=video, 2=other modality)

        Returns:
            modality_indices: [B, L_pad] long tensor, padded per tokenizer padding side
        """
        # Use the original attention mask to derive per-sample valid sequence lengths
        if attention_mask is None:
            attn_mask_bool = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attn_mask_bool = attention_mask.bool()

        modality_indices_list = []
        cur_feature_idx = 0
        for batch_idx in range(input_ids.shape[0]):
            # Extract the unpadded 1D sequence for this sample
            cur_input_ids = input_ids[batch_idx][attn_mask_bool[batch_idx]]

            # Count image special tokens (these mark where features are inserted)
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum().item()

            # If no image token, entire sequence is text
            if num_images == 0:
                modality_indices_list.append(torch.zeros((cur_input_ids.shape[0],), device=input_ids.device, dtype=torch.long))
                # Keep index movement consistent with embedding assembly logic
                cur_feature_idx += 1
                continue

            # Split around image token positions to obtain text segment lengths
            image_token_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            image_token_indices = [-1] + image_token_positions + [cur_input_ids.shape[0]]
            text_chunk_lengths = [int(image_token_indices[i+1] - image_token_indices[i] - 1) for i in range(len(image_token_indices)-1)]

            # Interleave text (0) and feature token types (1=image, 2=video)
            parts = []
            for i in range(num_images + 1):
                # text segment
                t_len = text_chunk_lengths[i]
                if t_len > 0:
                    # Mark the initial prompt tokens (before the first IMAGE_TOKEN_INDEX) as -1
                    if i == 0:
                        parts.append(torch.full((t_len,), -1, device=input_ids.device, dtype=torch.long))
                    else:
                        parts.append(torch.zeros((t_len,), device=input_ids.device, dtype=torch.long))
                # feature segment (after each image token)
                if i < num_images:
                    feat_types = feature_token_types[cur_feature_idx]
                    parts.append(feat_types.to(device=input_ids.device, dtype=torch.long))
                    cur_feature_idx += 1

            modality_indices_list.append(torch.cat(parts) if len(parts) > 0 else torch.zeros((0,), device=input_ids.device, dtype=torch.long))

        # Optionally truncate to tokenizer max length to mirror embedding truncation
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            modality_indices_list = [x[:tokenizer_model_max_length] for x in modality_indices_list]

        # Pad to a common length per tokenizer padding side; use -1 for padding
        max_len = max((int(x.shape[0]) for x in modality_indices_list), default=0)
        modality_indices = torch.full((input_ids.shape[0], max_len), -1, device=input_ids.device, dtype=torch.long)
        padding_side = getattr(self.config, 'tokenizer_padding_side', 'right')
        for i, seq in enumerate(modality_indices_list):
            cur_len = int(seq.shape[0])
            if cur_len == 0:
                continue
            if padding_side == 'left':
                modality_indices[i, -cur_len:] = seq
            else:
                modality_indices[i, :cur_len] = seq

        return modality_indices

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
                # for the image frames
                images=None,
                image_sizes=None,
                modalities=None,
                video_metas=None,
            ):
        """
        Args:
            input_ids: torch.Tensor of shape [B, text_len]
                Instruction text ids, e.g., torch.Size([4, 80])
            labels: torch.Tensor of shape [B, text_len]
                Answer text ids, e.g., torch.Size([4, 80])
            video_feats: torch.Tensor with shape [B, C, T, H, W]
                Other modality feature in video sequence, e.g., torch.Size([4, 3, 340, 256, 256])
            video_feat_fps: torch.Tensor of shape [B, ]
                Frame rate of the video sequence, e.g., torch.Size([4])
            feat_frame_nums: torch.Tensor of shape [B, ]
                Frame number of the video sequence, e.g., torch.Size([4])
            question_ids: torch.Tensor of shape [B, Max_seq_len]
                Question embedding in the instruction pairs, e.g., torch.Size([4])
            question_lens: torch.Tensor of shape [B, ]
                Question length in the instruction pairs, e.g., torch.Size([4])
            images: List[torch.Tensor]
                Each of shape torch.Size([num_frames, image_tokens_per_frame, hidden_size_form_vision_tower])
            image_sizes: list[int]
                Represents the H*W*C of the original video frames
            #TODO: this should have more modality support, like audio, depth map, etc.
            modalities: list
                Length=batch_size, each element is 'video'
            video_metas: Optional
                Additional video metadata
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif attention_mask is not None:
            batch_size = attention_mask.shape[0]
        elif isinstance(video_feats, list):
            batch_size = len(video_feats)
        elif video_feats is not None and hasattr(video_feats, "shape"):
            batch_size = int(video_feats.shape[0])
        elif isinstance(images, list):
            batch_size = len(images)
        elif images is not None and hasattr(images, "shape"):
            batch_size = int(images.shape[0])
        else:
            batch_size = 1

        video_tower = self.get_video_tower()
        vision_tower = self.get_vision_tower()

        if (video_tower is None and vision_tower is None) or \
            (video_feats is None and images is None) or \
            input_ids.shape[1] == 1: # this could be used for the inference
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
        
        # embed the question id using our LLM
        if question_ids is not None:
            question_embeds = self.get_model().embed_tokens(question_ids).detach()
        else:
            question_embeds = None
        
        modalities_per_sample = self._normalize_modalities_per_sample(modalities, batch_size)

        # figure out the chunk size
        if images is not None:
            chunk_num = images[0].shape[0]
        else:
            chunk_num = None

        batch_size = input_ids.shape[0]
        # get the video sequence image feature (The slow feature)
        # ipdb.set_trace() # check feature before mlp
        # Keep the return of using the images feature before mlp
        self.use_slow_feat_before_mlp = True
        if images is not None and (-1 not in image_sizes):
            image_features, video_idx_in_batch, image_features_before_mlp, scaled_shape = self.prepare_image_features(images, image_sizes, modalities_per_sample, return_feat_before_mlp=True) # [torch.Size([32, 196, 896]), torch.Size([32, 196, 896])]
        else: # the image feature is loaded
            # ipdb.set_trace() # test the handle of the image feature
            if sum(image_sizes) != -len(image_sizes):
                raise ValueError(f"Invalid image_sizes: expected all elements to be -1, but got {image_sizes}")
            image_features = images
            video_idx_in_batch = []
            for idx, sample_modalities in enumerate(modalities_per_sample):
                if "video" in sample_modalities:
                    video_idx_in_batch.append(idx)

        # Compute the relative time for each frame in each video sequence
        # video_feat_fps: List[int], feat_num_frames: torch.Tensor of shape [B,]
        # Output: relative_times: List[torch.Tensor], each of shape [num_frames] for each video in batch
        relative_times = []
        for i, (fps, num_frames) in enumerate(zip(video_feat_fps, feat_frame_nums)):
            # Generate frame indices: 0, 1, ..., num_frames-1
            frame_indices = torch.arange(num_frames, device=feat_frame_nums.device, dtype=torch.float32)
            # Compute time in seconds for each frame
            if float(fps) <= 0:
                raise ValueError(f"Invalid fps value: {fps} at index {i}. FPS must be positive.")
            times = frame_indices / float(fps)
            # Normalize to [0, 1] relative time (0=start, 1=end)
            if num_frames > 1:
                rel_time = times / times[-1]
            else:
                rel_time = torch.zeros_like(times)
            relative_times.append(rel_time)
    
        if video_feats is not None and not isinstance(video_feats, list):
            video_features = [video_feats[i][:, :feat_frame_nums[i]].permute(1, 2, 3, 0).contiguous() for i in range(len(video_feats))] # (B, C, T, H, W) -> B * [T, H * W, C]
        elif video_feats is not None and isinstance(video_feats, list):
            video_features = video_feats
        else:
            video_features = None
        
        if images is not None:
            image_features = self.post_processing_of_image_feature(image_features, video_idx_in_batch) # [torch.Size([6720, 896]), torch.Size([6720, 896])]
            
        # Do the very first version by concatenating the feature
        feat_combine_method = getattr(self.config, 'feat_combine_method', 'concat')        
        # Track per-sample feature token types for modality indices (1=image, 2=video)
        feature_token_types = []
        if video_features is not None:
            for batch_idx, (curr_video_feat, curr_image_feat) in enumerate(zip(video_features, image_features)):
                curr_video_feat = self.input_mapping(curr_video_feat)

                # Prepend learned modality tokens before each modality span (image=1, video=2)
                # Use precomputed `modality_tokens` aligned with `modalities[batch_idx]`
                add_image_token_type = False
                add_video_token_type = False

                # Normalize to list for indexing
                curr_modalities = modalities_per_sample[batch_idx]

                # Video token
                if 'video' in curr_modalities:
                    video_modality_token = self.modality_tokens
                    video_modality_token = video_modality_token.to(device=curr_image_feat.device, dtype=curr_image_feat.dtype)
                    curr_image_feat_with_token = torch.cat([video_modality_token, curr_image_feat], dim=0)
                    add_image_token_type = True
                else:
                    curr_image_feat_with_token = curr_image_feat

                # Other modalities like audio, 3d_feature
                if 'audio' in curr_modalities or 'dense_video' in curr_modalities:
                    curr_video_feat = rearrange(curr_video_feat, "t h w d -> (t h w) d")
                    audio_modality_token = self.modality_tokens
                    audio_modality_token = audio_modality_token.to(device=curr_video_feat.device, dtype=curr_video_feat.dtype)
                    curr_video_feat_with_token = torch.cat([audio_modality_token, curr_video_feat], dim=0)
                    add_video_token_type = True
                elif '3d_feature' in curr_modalities:
                    three_d_feature_modality_token = self.modality_tokens
                    curr_video_feat = self.image_shape_alignment(curr_video_feat, scaled_shape)

                    if feat_combine_method == 'add':
                        curr_video_feat = rearrange(curr_video_feat, "t h w d -> t (h w) d")
                        curr_video_feat = self.post_processing_of_image_feature([curr_video_feat], video_idx_in_batch)[0]
                    elif feat_combine_method == 'concat':
                        curr_video_feat = rearrange(curr_video_feat, "t h w d -> (t h w) d")
                    else:
                        raise ValueError(f"Invalid feat_combine_method: {feat_combine_method}")

                    three_d_feature_modality_token = three_d_feature_modality_token.to(device=curr_video_feat.device, dtype=curr_video_feat.dtype)
                    curr_video_feat_with_token = torch.cat([three_d_feature_modality_token, curr_video_feat], dim=0)
                    add_video_token_type = True
                else:
                    curr_video_feat = rearrange(curr_video_feat, "t h w d -> (t h w) d")
                    curr_video_feat_with_token = curr_video_feat

                # Record token-type split including modality tokens
                img_len = curr_image_feat.shape[0]
                vid_len = curr_video_feat.shape[0]

                type_segments = []
                if add_image_token_type:
                    # Add type for all modality special tokens (not just 1)
                    type_segments.append(torch.full((self.modality_token_num,), 1, device=curr_image_feat.device, dtype=torch.long))
                type_segments.append(torch.full((img_len,), 1, device=curr_image_feat.device, dtype=torch.long))
                if add_video_token_type:
                    # Add type for all modality special tokens (not just 1)
                    type_segments.append(torch.full((self.modality_token_num,), 2, device=curr_video_feat.device, dtype=torch.long))
                type_segments.append(torch.full((vid_len,), 2, device=curr_video_feat.device, dtype=torch.long))
                feature_token_types.append(torch.cat(type_segments, dim=0))
                
                combined_feat = torch.cat([curr_image_feat_with_token, curr_video_feat_with_token], dim=0)
                video_features[batch_idx] = combined_feat

            new_frame_num = [ele.shape[0] for ele in video_features]
        else: # there are only image features or RGB image video sequence
            # ipdb.set_trace() # the postprocessing
            video_features = image_features
            for ele in video_features: # add grad to the forward pass
                ele.requires_grad = True
            new_frame_num = [ele.shape[0] for ele in image_features]
            # Only image tokens in this case
            for ele in video_features:
                feature_token_types.append(torch.full((ele.shape[0],), 1, device=ele.device, dtype=torch.long))

        # Assemble multimodal embeddings and reconcile optional inputs
        orig_attention_mask = attention_mask
        new_input_embeds, new_labels, attention_mask, position_ids = self._assemble_text_and_feature_embeddings(
            input_ids,
            labels,
            attention_mask,
            position_ids,
            video_features,
            new_frame_num,
        )

        # Ensure gradients can flow through inputs_embeds when training under DeepSpeed/gradient checkpointing
        if self.training and new_input_embeds is not None and not new_input_embeds.requires_grad:
            new_input_embeds.requires_grad_(True)

        # Build per-token modality indices aligned with `new_input_embeds`
        modality_indices = self._build_modality_indices(
            input_ids=input_ids,
            attention_mask=orig_attention_mask,
            feature_token_types=feature_token_types,
        )
        # Save for optional downstream inspection
        self.latest_modality_indices = modality_indices

        # Step 1: Aggregate modality tokens into modality-special tokens via cross-attention
        new_input_embeds = self.modality_special_token_aggregator(new_input_embeds, 
                                                                  modality_indices)
        
        # Step 2: Instruction-modality interaction to balance cross-modal information
        new_input_embeds = self.modality_aggregator(hidden_states=new_input_embeds, 
                                                    attention_mask=attention_mask,
                                                    position_ids=position_ids,
                                                    modality_indices=modality_indices)
                                                    
        # Fuse fast 3D tokens into slow video tokens using modality indices, then drop fused fast tokens
        if any(('3d_feature' in m) for m in modalities_per_sample) and feat_combine_method == 'add':
            B, _, H = new_input_embeds.shape
            keep_lengths = torch.zeros(B, dtype=torch.long, device=attention_mask.device)

            # Pass 1: fuse and record kept lengths
            for b in range(B):
                vid_idx = torch.nonzero(modality_indices[b] == 1, as_tuple=False).flatten()
                d3_idx = torch.nonzero(modality_indices[b] == 2, as_tuple=False).flatten()
                if vid_idx.numel() > 0 and d3_idx.numel() > 0:
                    if vid_idx.numel() != d3_idx.numel():
                        raise ValueError(f"Fusion mismatch: video tokens ({vid_idx.numel()}) != 3D tokens ({d3_idx.numel()})")
                    new_input_embeds[b, vid_idx] += new_input_embeds[b, d3_idx]
                    attention_mask[b, d3_idx] = 0
                    modality_indices[b, d3_idx] = -1
                keep_lengths[b] = attention_mask[b].sum()

            max_len = int(keep_lengths.max().item()) if B > 0 else 0
            out_embeds = new_input_embeds.new_zeros((B, max_len, H))
            out_masks = attention_mask.new_zeros((B, max_len))
            out_mods = modality_indices.new_full((B, max_len), -1)
            out_pos = position_ids.new_zeros((B, max_len)) if position_ids is not None else None
            out_labels = new_labels.new_full((B, max_len), -100) if new_labels is not None else None

            # Pass 2: compact per-batch and pad
            for b in range(B):
                keep = attention_mask[b].bool()
                Lb = int(keep_lengths[b].item())
                if Lb == 0:
                    continue
                out_embeds[b, :Lb] = new_input_embeds[b][keep]
                out_masks[b, :Lb] = attention_mask[b][keep]
                out_mods[b, :Lb] = modality_indices[b][keep]
                if out_pos is not None:
                    out_pos[b, :Lb] = position_ids[b][keep]
                if out_labels is not None:
                    out_labels[b, :Lb] = new_labels[b][keep]

            new_input_embeds = out_embeds
            attention_mask = out_masks
            modality_indices = out_mods
            if out_pos is not None:
                position_ids = out_pos
            if out_labels is not None:
                new_labels = out_labels
                          
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, modality_indices

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        video_feats: Optional[torch.Tensor] = None,
        video_feat_fps: Optional[torch.FloatTensor] = None,
        feat_frame_nums: Optional[torch.FloatTensor] = None,

        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        modalities: Optional[List[str]] = None,        
        
        question_ids = None,
        question_lens = None,    
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        # import ipdb
        # ipdb.set_trace()
        if video_feats is not None or images is not None: # if there is vision input
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                video_feats,
                video_feat_fps=video_feat_fps,
                feat_frame_nums=feat_frame_nums,
                question_ids=question_ids,
                question_lens=question_lens,
                images=images,
                image_sizes=image_sizes,
                modalities=modalities,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # Ensure gradients for generation-time custom embeddings when needed (safe no-op in eval)
        if inputs_embeds is not None and not inputs_embeds.requires_grad and self.training:
            inputs_embeds.requires_grad_(True)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.half(),
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        ## TODO: which function will call this function? What should we change here? 
        videos = kwargs.pop("videos", None)
        video_sizes = kwargs.pop("video_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if videos is not None:
            inputs['videos'] = videos
        if video_sizes is not None:
            inputs['video_sizes'] = video_sizes
        return inputs

class UniMVUModalityAggregator(nn.Module):
    """
    UniMVUModalityAggregator is the modality aggregator for UniMVU with Qwen2.
    Applies instruction-to-modality cross-attention to balance multimodal information.
    """
    def __init__(self, hidden_size=896, num_heads=14, num_key_value_heads=14, 
                 rope_theta=250000, attention_dropout=0.0, modality_token_num=1):
        super(UniMVUModalityAggregator, self).__init__()
        self.hidden_size = hidden_size
        self.self_attn = UniMVUCrossModalityInteractionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            rope_theta=rope_theta,
            attention_dropout=attention_dropout,
            modality_token_num=modality_token_num
        )
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor]= None,
    ) -> torch.Tensor:
        """
        Apply instruction-modality interaction.
        
        Args:
            hidden_states: [B, L, D] input embeddings
            modality_indices: [B, L] modality type per token (0=text, 1=image, 2=video, -1=ignore)
            attention_mask: optional attention mask
            position_ids: optional position ids
            
        Returns:
            hidden_states: [B, L, D] output embeddings after cross-modal interaction
        """
        residual = hidden_states
        original_hidden_states = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Cross-modality attention (instruction queries attend to modality keys)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            modality_indices=modality_indices,
        )

        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        outputs = residual + hidden_states

        if modality_indices is not None:
            preserve_mask = (modality_indices == -1) | (modality_indices == 0)
            if preserve_mask.any():
                outputs = torch.where(preserve_mask.unsqueeze(-1), original_hidden_states, outputs)
        
        return outputs


class UniMVUCrossModalityInteractionLayer(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self,
        hidden_size=896,
        num_heads=14,
        num_key_value_heads=14,
        rope_theta=250000,
        attention_dropout=0.0,
        modality_token_num=0,   
        ):
        super(UniMVUCrossModalityInteractionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.modality_token_num = modality_token_num

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim, base=self.rope_theta)

    def _compute_modality_level_scores(
        self,
        attn_weights: torch.Tensor,
        modality_indices: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute modality-level importance scores based on instruction→special token attention.
        
        Strategy:
        1. Extract attention weights between instruction tokens and modality-special tokens
        2. Normalise the instruction attention mass across the available modalities per sample
        
        Args:
            attn_weights: [B, H, L, L] post-softmax attention (queries x keys)
            modality_indices: [B, L] modality type per token (0=instruction, 1=image, 2=video, -1=ignore)
            
        Returns:
            modality_scores: [B, num_modalities] importance scores per modality or None if no modalities present
        """
        B = attn_weights.shape[0]

        # Find all modalities (excluding -1 for padding and 0 for instruction)
        all_modalities = sorted(list(set(modality_indices.view(-1).tolist()) - {-1, 0}))
        num_modalities = len(all_modalities)

        if num_modalities == 0:
            return None

        device = attn_weights.device
        dtype = attn_weights.dtype
        modality_scores = torch.zeros(B, num_modalities, device=device, dtype=dtype)
        modality_present = torch.zeros(B, num_modalities, device=device, dtype=torch.bool)

        instruction_positions = [torch.where(modality_indices[b] == 0)[0] for b in range(B)]

        for mod_idx, m in enumerate(all_modalities):
            modality_mask = (modality_indices == m)
            modality_present[:, mod_idx] = modality_mask.any(dim=-1)
            if not modality_present[:, mod_idx].any():
                continue

            for batch_idx in range(B):
                if not modality_present[batch_idx, mod_idx]:
                    continue

                inst_pos = instruction_positions[batch_idx]
                if inst_pos.numel() == 0:
                    continue

                modality_positions = torch.where(modality_mask[batch_idx])[0]
                if modality_positions.numel() == 0:
                    continue

                special_token_positions = modality_positions[:self.modality_token_num]
                if special_token_positions.numel() == 0:
                    continue

                inst_to_special = attn_weights[batch_idx, :, inst_pos[:, None], special_token_positions]
                score = inst_to_special.float().mean()
                modality_scores[batch_idx, mod_idx] = score.to(dtype)

        if not modality_present.any():
            return None

        normalized_scores = torch.zeros_like(modality_scores)
        score_sums = modality_scores.float().sum(dim=-1, keepdim=True)

        valid_rows = (score_sums.squeeze(-1) > 0)
        if valid_rows.any():
            normalized_scores[valid_rows] = (
                modality_scores[valid_rows].float() / score_sums[valid_rows]
            ).to(dtype)

        fallback_rows = (~valid_rows) & modality_present.any(dim=-1)
        if fallback_rows.any():
            for batch_idx in torch.where(fallback_rows)[0].tolist():
                available = modality_present[batch_idx]
                count = int(available.sum().item())
                if count > 0:
                    normalized_scores[batch_idx, available] = normalized_scores.new_full((count,), 1.0 / count)

        normalized_scores *= modality_present.to(normalized_scores.dtype)
        # print(normalized_scores)
        # print("--------------------------------")
        non_modal_rows = ~modality_present.any(dim=-1)
        if non_modal_rows.any():
            normalized_scores[non_modal_rows] = 0.0

        return normalized_scores

    def _compute_cross_modality_attention(
        self,
        attn_weights: torch.Tensor,
        attn_output: torch.Tensor,
        modality_indices: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], List[int]]:
        """
        Reuse the existing self-attention distribution to build instruction-driven gains for each modality.

        Args:
            attn_weights: [B, H, L, L] post-softmax attention (queries x keys)
            attn_output: [B, H, L, D] contextualised values after self-attention
            modality_indices: [B, L] modality id per token (0=text, >0 modality, -1 ignored)

        Returns:
            per_modality_outputs: dict mapping modality id -> [B, L, H, D] contribution tensor
            all_modalities: sorted list of modality ids with content
        """
        B, H, L, D = attn_output.shape
        attn_output_t = attn_output.transpose(1, 2).contiguous()  # (B, L, H, D)

        # Collect instruction queries (0) and enumerate non-text modalities
        instruction_mask = (modality_indices == 0)
        all_modalities = sorted(list(set(modality_indices.view(-1).tolist()) - {-1, 0}))

        per_modality_outputs: Dict[int, torch.Tensor] = {}
        if not all_modalities:
            return per_modality_outputs, all_modalities

        eps = 1e-6

        if instruction_mask.any():
            attn_from_instructions = attn_weights * instruction_mask.unsqueeze(1).unsqueeze(-1).to(attn_weights.dtype)
            token_importance = attn_from_instructions.sum(dim=2).mean(dim=1)  # (B, L)
        else:
            token_importance = attn_weights.new_zeros(B, L)

        for m_index in all_modalities:
            modality_mask = (modality_indices == m_index)
            if not modality_mask.any():
                continue

            modality_mask_f = modality_mask.to(token_importance.dtype)
            importance_m = token_importance * modality_mask_f

            token_count = modality_mask_f.sum(dim=-1, keepdim=True)
            total_importance = importance_m.sum(dim=-1, keepdim=True)

            uniform = torch.where(
                token_count > 0,
                modality_mask_f / token_count.clamp_min(1.0),
                torch.zeros_like(importance_m),
            )

            normalized = torch.where(
                total_importance > eps,
                importance_m / total_importance.clamp_min(eps),
                uniform,
            )

            # delta = (normalized - uniform) * modality_mask_f  # (B, L), zero outside modality span
            delta = normalized * modality_mask_f
            mod_output = attn_output_t * delta.unsqueeze(-1).unsqueeze(-1)
            per_modality_outputs[m_index] = mod_output

        return per_modality_outputs, all_modalities

    def _merge_attention_outputs(
        self,
        attn_output: torch.Tensor,
        modality_balanced_attention_output: torch.Tensor,
        modality_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return attn_output + modality_balanced_attention_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        modality_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Cross-modality interaction layer: modality tokens attend to instruction tokens.
        
        Args:
            hidden_states: [B, L, D]
            modality_indices: [B, L] where 0=instruction, 1=image, 2=video, -1=padding/prompt
            
        Returns:
            attn_output: [B, L, D] with added cross-modal attention for modality tokens
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Normalize attention_mask shape without rebuilding when passed from parent
        if attention_mask.size() != (bsz, q_len):
            raise ValueError(f"Attention mask should be of size {(bsz, q_len)}, but is {attention_mask.size()}")
        attention_mask = _prepare_4d_attention_mask(attention_mask, attn_weights.dtype, q_len)
        
        if modality_indices is not None:
            min_dtype = torch.finfo(attn_weights.dtype).min
            # 1. Instruction queries (0) mask out Text keys (0, -1). Allow Modality keys (>0).
            is_instruction_query = (modality_indices == 0)
            is_text_key = (modality_indices == 0) | (modality_indices == -1)
            instruction_mask = is_instruction_query.unsqueeze(-1) & is_text_key.unsqueeze(1)

            # 2. Prompt queries (-1) mask out ALL keys.
            is_prompt_query = (modality_indices == -1)
            prompt_mask = is_prompt_query.unsqueeze(-1) # Broadcasts to mask entire row

            # Combine masks
            mask_condition = instruction_mask | prompt_mask
            
            # Apply mask [B, 1, L, L]
            attn_weights = attn_weights.masked_fill(mask_condition.unsqueeze(1), min_dtype)

        attn_weights = attn_weights + attention_mask
        
        # Standard self-attention
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Cross-modality interaction with modality-level balancing
        if modality_indices is not None:  # Skip during generation (single token)
            # Step 1: Compute intra-modality cross-attention (modality→instruction, using contextualized outputs)
            per_modality_outputs, all_modalities = self._compute_cross_modality_attention(
                attn_weights, attn_output, modality_indices
            )
            
            # Step 2: Compute modality-level importance scores
            modality_scores = self._compute_modality_level_scores(
                attn_weights, modality_indices
            )
            # modality_scores = self._compute_modality_level_scores(
            #     attn_weights_before_softmax, modality_indices
            # )

            # Step 3: Reweight per-modality outputs by their importance scores
            modality_balanced_attention_output = torch.zeros(bsz, q_len, self.num_heads, self.head_dim, 
                                                            device=attn_output.device, dtype=attn_output.dtype)
            
            if modality_scores is not None:
                for mod_idx, m in enumerate(all_modalities):
                    # Get this modality's output: [B, L, H, D]
                    mod_output = per_modality_outputs[m]
                    # Get this modality's score: [B,] (broadcast to [B, 1, 1, 1])
                    score = modality_scores[:, mod_idx].view(bsz, 1, 1, 1)
                    # Reweight and accumulate
                    modality_balanced_attention_output += mod_output * score
        else:
            modality_balanced_attention_output = torch.zeros(bsz, q_len, self.num_heads, self.head_dim,
                                                            device=attn_output.device, dtype=attn_output.dtype)

        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, L, H, D)
        attn_output = self._merge_attention_outputs(
            attn_output,
            modality_balanced_attention_output,
            modality_indices,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output
        

class ModalityTokenAggregator(nn.Module):
    """
    Aggregates modality spans so that each modality's special token becomes a
    weighted summary over all tokens in that modality (per sample, batched).

    Learnable components:
    - q/k/v projections to compute attention in a learned space
    - logit_scale to adjust attention temperature
    - residual_gate to blend summary with the original special token
    - pre-attention LayerNorm for stability
    """
    def __init__(self, hidden_size: int, modality_token_num: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.residual_gate = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(hidden_size)
        self.modality_token_num = modality_token_num

    def forward(self, hidden_states: torch.Tensor,
                modality_indices: Optional[torch.Tensor]) -> torch.Tensor:
        if modality_indices is None:
            return hidden_states

        batch_size, seq_len, dim = hidden_states.shape
        if modality_indices.shape[1] < seq_len:
            return hidden_states

        updated_hidden = hidden_states.clone()
        normed = self.norm(hidden_states)
        base_scale = 1.0 / math.sqrt(float(dim))
        temperature = torch.exp(self.logit_scale)
        gate = torch.sigmoid(self.residual_gate)

        for b in range(batch_size):
            cur_mod_idx = modality_indices[b, :seq_len]
            present_modalities = torch.unique(cur_mod_idx)
            present_modalities = present_modalities[present_modalities > 0]
            if present_modalities.numel() == 0:
                continue
            for m in present_modalities.tolist():
                token_positions = torch.nonzero(cur_mod_idx == m, as_tuple=False).view(-1)
                if token_positions.numel() == 0:
                    continue
                # First N positions in the span are the special tokens (queries)
                special_positions = token_positions[0:self.modality_token_num]  # [modality_token_num]
                if special_positions.numel() < self.modality_token_num:
                    raise ValueError(f"Modality {m} has only {special_positions.numel()} tokens, but expected at least {self.modality_token_num} special tokens")
                
                # Project special tokens as queries
                q = self.q_proj(normed[b, special_positions, :])  # [modality_token_num, D]
                
                # All modality tokens (including special tokens) as keys/values
                k = self.k_proj(normed[b, token_positions, :])   # [N, D]
                v = self.v_proj(normed[b, token_positions, :])   # [N, D]

                # Compute attention: [modality_token_num, N]
                attn_logits = torch.matmul(q, k.transpose(0, 1)) * (base_scale * temperature)
                attn_weights = torch.softmax(attn_logits, dim=-1)  # [modality_token_num, N]
                
                # Aggregate: [modality_token_num, D]
                summarized = torch.matmul(attn_weights, v)

                # Update all special tokens with gated residual (vectorized)
                original = hidden_states[b, special_positions, :]  # [modality_token_num, D]
                updated_hidden[b, special_positions, :] = gate * summarized + (1.0 - gate) * original

        return updated_hidden

AutoModelForCausalLM.register(UniMVUQwen2Config, UniMVUQwen2ForCausalLM)
