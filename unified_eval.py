import argparse
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

# Suppress the harmless meta device warning during model loading
warnings.filterwarnings('ignore', message='.*copying from a non-meta parameter.*to a meta parameter.*')

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import AutoTokenizer, BitsAndBytesConfig

# Project imports (training-consistent helpers)
from libs.conversation_lib import conv_templates, SeparatorStyle
from libs.mm_utils import KeywordsStoppingCriteria, tokenizer_vision_token
from libs.dataset.eval_datasets import (
    AVQADataset,
    AVSDDataset,
    MusicAVQADataset,
    ScanQADataset,
    SQA3DDataset,
)
from libs.model.multimodal_LMM._unimvu_base import (
    UniMVUQwen2Config,
    UniMVUQwen2ForCausalLM,
)
from libs.model.multimodal_LMM._unimvu_mix import (
    UniMVUUnifiedConfig,
    UniMVUUnifiedForCausalLM,
)
from libs.utils.train_utils import MODEL_ARGUMENTS_MAPPING
# Import decord AFTER libs.model to avoid segmentation fault due to library conflicts
try:
    import decord  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    decord = None

# Constants (align with existing evaluators)
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


DatasetName = Literal["avqa", "avsd", "music_avqa", "scanqa", "sqa3d"]
ModelKind = Literal["unimvu", "unimvu_uni"]

BASE_MODEL_TYPE = "unimvu"
UNIFIED_MODEL_TYPE = "unimvu_uni"


def prepare_audio_feature(
    audio_feature_raw: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, int]:
    """Normalize audio feature tensors into model-expected shape.

    Supports both ImageBind-style 3D and LanguageBind-style 4D inputs, matching
    the training-time shaping logic used across existing evaluators.

    Args:
        audio_feature_raw: Input tensor with shape (B, T, D) or (B, H, W, D) or similar.
        device: Target device for the tensor.
        dtype: Target dtype for the tensor.
    Returns:
        A tuple of (audio_feature, feat_frame_num) where audio_feature has shape
        (B, C, T, 1, 1) as expected by the model, and feat_frame_num is the T length.
    """
    if audio_feature_raw is None or not hasattr(audio_feature_raw, "shape"):
        zeros = torch.zeros((1, 10, 1024), dtype=dtype, device=device)
        return zeros, int(zeros.shape[1])
    if len(audio_feature_raw.shape) == 2:
        audio_feature_raw = audio_feature_raw.permute([1,0]) # (C, T)
    # unsqueeze dim 
        audio_feature = audio_feature_raw.unsqueeze(dim=-1).unsqueeze(dim=-1) # (C, T, 1, 1) (C, T, H, W)
        return audio_feature, int(audio_feature.shape[1])

    if len(audio_feature_raw.shape) == 3:
        # ImageBind feature: (B, T, D) -> (B, D, T, 1, 1)
        audio_feature = (
            audio_feature_raw.permute(0, 2, 1)  # (B, D, T)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .to(device=device, dtype=dtype)
        )
        return audio_feature, int(audio_feature.shape[2])

    if len(audio_feature_raw.shape) == 4:
        # LanguageBind feature: (B, H, W, D) => view to (B, T, D) with T=H*W
        feat_dim = int(audio_feature_raw.shape[-1])
        temp_bs = int(audio_feature_raw.shape[0])
        if temp_bs == 1:
            viewed = audio_feature_raw.view(temp_bs, -1, feat_dim)
            audio_feature = (
                viewed.permute(0, 2, 1)  # (B, D, T)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to(device=device, dtype=dtype)
            )
            return audio_feature, int(audio_feature.shape[2])

    zeros = torch.zeros((1, 10, 1024), dtype=dtype, device=device)
    return zeros, int(zeros.shape[1])

def load_trained_model_for_eval(model_path,
                                model_base,
                                model_arg_name='VideoFeatModelArgumentsUniMVU',
                                data_arg_name='default',
                                load_8bit=False, load_4bit=False,
                                device_map="auto", device="cuda",
                                use_flash_attn=False,
                                model_type: str = BASE_MODEL_TYPE,
                                **kwargs):
    # Force single-device placement to avoid mixed-device submodules
    if device == "cuda":
        kwargs = {"device_map": {"": 0}, **kwargs}
        load_to_cpu = False
    else:
        load_to_cpu = True

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16 if not load_to_cpu else torch.float32

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    # Load tokenizer from base to avoid custom config tokenizer mapping issues
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

    requested_model_type = (model_type or BASE_MODEL_TYPE).lower()

    if requested_model_type == UNIFIED_MODEL_TYPE:
        ConfigCls, ModelCls = UniMVUUnifiedConfig, UniMVUUnifiedForCausalLM
    elif requested_model_type == BASE_MODEL_TYPE:
        ConfigCls, ModelCls = UniMVUQwen2Config, UniMVUQwen2ForCausalLM
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Use '{BASE_MODEL_TYPE}' or '{UNIFIED_MODEL_TYPE}'."
        )

    # Merge training-time config with default model args
    default_model_arg = MODEL_ARGUMENTS_MAPPING[model_arg_name]

    print(
        "Loading model from base with class:",
        ModelCls.__name__,
        "requested_model_type:",
        requested_model_type,
    )
    lora_cfg_pretrained = ConfigCls.from_pretrained(model_path)
    # Test
    lora_cfg_pretrained.mm_video_tower = None
    model = ModelCls.from_pretrained(
        model_base,
        low_cpu_mem_usage=not load_to_cpu,
        config=lora_cfg_pretrained,
        **kwargs,
    )
    print('Warning: using MODEL_ARGUMENTS_MAPPING:', model_arg_name, 'DATA_ARGUMENTS_MAPPING:', data_arg_name)
    for key in default_model_arg.__dict__:
        if not key.startswith('__'):
            if not hasattr(lora_cfg_pretrained, key):
                setattr(lora_cfg_pretrained, key, default_model_arg.__dict__[key])

    target_vocab = len(tokenizer)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and input_embeddings.weight.size(0) != target_vocab:
        model.resize_token_embeddings(target_vocab)

    # Initialize vision modules
    if getattr(default_model_arg, 'video_tower', None) is not None:
        model.get_model().initialize_vision_modules(
            model_args=lora_cfg_pretrained,
            fsdp=None,
        )

    # Load LoRA weights and merge
    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')


    # Ensure model and vision tower reside on the same device
    target_device = torch.device('cuda') if device == 'cuda' else torch.device(device)
    model.to(target_device)
    vt = model.get_vision_tower()
    if vt is not None and hasattr(vt, 'vision_tower'):
        vt.vision_tower.to(target_device)

    # Initialize tokenizer special tokens and custom mappings
    model.initialize_vision_tokenizer(lora_cfg_pretrained, tokenizer=tokenizer)
    # Initialize any custom layers dependent on model_args (e.g., input_mapping)
    if hasattr(model, 'initialize_custom_config'):
        model.initialize_custom_config(lora_cfg_pretrained)
        # Re-sync devices and dtypes since new modules may have been created on CPU/float32
        model.to(target_device)
        model_dtype = next(model.parameters()).dtype
        if hasattr(model, 'input_mapping') and isinstance(model.input_mapping, torch.nn.Module):
            model.input_mapping.to(device=target_device, dtype=model_dtype)

    # Optionally load non-LoRA trainables with simple prefix strip + exact match
    non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        obj = torch.load(non_lora_path, map_location='cpu')
        src = obj.get('state_dict', obj) if isinstance(obj, dict) else obj
        if not isinstance(src, dict):
            print('non_lora_trainables is not a dict-like state_dict; skipping load.')
            src = {}
        tgt_state = model.state_dict()
        print(f"Found {len(src)} non-LoRA tensors in checkpoint.")
        keys_list = list(src.keys())
        print("non_lora_trainables keys:", keys_list)

        def normalize_key(k: str):
            for p in (
                'peft_model.base_model.model.',
                'peft_module.base_model.model.',
                'base_model.model.',
                'model.',
            ):
                if k.startswith(p):
                    return k[len(p):]
            return k

        remapped = {}
        loaded_mapping = []
        failed_original = []
        for k, v in src.items():
            nk = normalize_key(k)
            if nk in tgt_state and hasattr(v, 'shape') and v.shape == tgt_state[nk].shape:
                remapped[nk] = v
                loaded_mapping.append((k, nk))
            else:
                failed_original.append(k)

        if remapped:
            _, unexpected = model.load_state_dict(remapped, strict=False)
            print(f"Loaded {len(remapped)} non-LoRA weights.")
            if unexpected:
                print("unexpected non-LoRA weights:", unexpected)
        else:
            print('No compatible non-LoRA tensors found to load.')
        print("Unloaded keys:", failed_original)

    image_processor = None
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


@dataclass
class EvalArgs:
    dataset: DatasetName
    model_type: ModelKind
    model_path: str
    model_base: Optional[str]
    model_arg_name: Optional[str]
    annotation_file: Optional[str]
    question_file: Optional[str]
    video_folder: str
    feature_folder: str
    pred_save: str
    conv_mode: str
    num_workers: int
    for_get_frames_num: int
    data_sample_ratio: Optional[float]
    feat_type: str
    temperature: float
    top_p: Optional[float]
    num_beams: int
    mm_resampler_type: str
    mm_spatial_pool_stride: int
    mm_spatial_pool_out_channels: int
    mm_spatial_pool_mode: str
    mm_newline_position: str
    modalities: list

def _build_prompt_with_image_token(text: str, use_im_tokens: bool) -> str:
    if use_im_tokens:
        return (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        )
    return DEFAULT_IMAGE_TOKEN + "\n" + text


def _get_modalities_for_dataset(dataset: DatasetName, user_modalities: list = None) -> list:
    """Determine the appropriate modalities for a given dataset.
    
    Args:
        dataset: The dataset name
        user_modalities: Optional user-provided modalities to validate
        
    Returns:
        List of modalities appropriate for the dataset
    """
    # Define dataset-specific modality requirements
    dataset_modalities = {
        "scanqa": ["video", "3d_feature"],  # Uses video frames + 3D features, but modality is "video"
        "sqa3d": ["video", "3d_feature"],   # Uses video frames + 3D features, but modality is "video"
        "avqa": ["video", "audio"],
        "avsd": ["video", "audio"],
        "music_avqa": ["video", "audio"],
    }
    
    required_modalities = dataset_modalities.get(dataset, ["video"])
    # Warn if dataset is not recognized and no user modalities provided
    if dataset not in dataset_modalities and not user_modalities:
        print(f"Warning: Dataset '{dataset}' is not recognized. Using default modalities ['video']. "
              f"Consider specifying --modalities explicitly for custom datasets.")
    # If user provided modalities, validate and warn if inconsistent
    if user_modalities and user_modalities != required_modalities:
        print(f"Warning: Dataset '{dataset}' typically uses modalities {required_modalities}, "
              f"but user specified {user_modalities}. Using user-specified modalities.")
        return user_modalities
    
    return required_modalities


def evaluate(args: EvalArgs) -> None:
    # Logging setup
    log_dir = os.path.join(
        ".", "eval_output", args.dataset, "eval_log"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"eval_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        filename=log_filename, level=logging.INFO, format="%(asctime)s - %(message)s"
    )

    # Load model/tokenizer/processor
    tokenizer, model, image_processor, _ = load_trained_model_for_eval(
        model_type=args.model_type,
        model_path=args.model_path,
        model_base=args.model_base,
        model_arg_name=args.model_arg_name,
    )
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Pass-through configuration flags for model behavior consistency
    model.config.mm_newline_position = args.mm_newline_position
    model.config.mm_spatial_pool_mode = args.mm_spatial_pool_mode

    # Build dataset
    ds: Dataset
    if args.dataset == "avqa":
        if args.annotation_file is None:
            raise ValueError("--annotation-file is required for avqa")
        ds = AVQADataset(
            annotation_file=args.annotation_file,
            video_folder=args.video_folder,
            feat_folder=args.feature_folder,
            image_processor=image_processor,
            for_get_frames_num=args.for_get_frames_num,
        )
    elif args.dataset == "avsd":
        if args.annotation_file is None:
            raise ValueError("--annotation-file is required for avsd")
        ds = AVSDDataset(
            annotation_file=args.annotation_file,
            video_folder=args.video_folder,
            feat_folder=args.feature_folder,
            image_processor=image_processor,
            for_get_frames_num=args.for_get_frames_num,
        )
    elif args.dataset == "music_avqa":
        if args.annotation_file is None:
            raise ValueError("--annotation-file is required for music_avqa")
        ds = MusicAVQADataset(
            annotation_file=args.annotation_file,
            video_folder=args.video_folder,
            feature_folder=args.feature_folder,
            image_processor=image_processor,
            feat_type=(args.feat_type or "audio"),
            for_get_frames_num=args.for_get_frames_num,
        )

    elif args.dataset == "scanqa":
        if args.question_file is None:
            raise ValueError("--question-file is required for scanqa")
        ds = ScanQADataset(
            question_file=args.question_file,
            video_folder=args.video_folder,
            feature_folder=args.feature_folder,
            image_processor=image_processor,
        )
    elif args.dataset == "sqa3d":
        if args.question_file is None:
            raise ValueError("--question-file is required for sqa3d")
        ds = SQA3DDataset(
            question_file=args.question_file,
            video_folder=args.video_folder,
            feature_folder=args.feature_folder,
            image_processor=image_processor,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_size = len(ds)
    if dataset_size == 0:
        raise ValueError(
            f"No valid samples found for dataset '{args.dataset}'. "
            "Check annotation/question files and feature directories."
        )

    if args.data_sample_ratio is not None:
        if args.data_sample_ratio < 0:
            raise ValueError("--data-sample-ratio must be non-negative.")
        requested_ratio = args.data_sample_ratio
        effective_ratio = min(requested_ratio, 1.0)
        if requested_ratio > 1.0:
            logging.warning(
                "data_sample_ratio %.4f exceeds 1.0; evaluating full dataset instead.",
                requested_ratio,
            )
        target_size = dataset_size
        if effective_ratio < 1.0:
            target_size = max(1, int(dataset_size * effective_ratio))
        if target_size < dataset_size:
            sampled_indices = sorted(random.sample(range(dataset_size), target_size))
            ds = Subset(ds, sampled_indices)
            logging.info(
                "Applied data_sample_ratio=%.4f -> evaluating %d/%d samples (%.4f effective ratio).",
                requested_ratio,
                target_size,
                dataset_size,
                target_size / dataset_size,
            )
            print(
                f"Applying data_sample_ratio={requested_ratio:.4f}: "
                f"evaluating {target_size}/{dataset_size} samples "
                f"(effective ratio {target_size / dataset_size:.4f})."
            )

    # DataLoader
    dl = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=None
    )

    # Determine appropriate modalities for the dataset
    effective_modalities = _get_modalities_for_dataset(args.dataset, args.modalities)
    print(f"Using modalities for {args.dataset}: {effective_modalities}")

    # Generation args
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": 1024,
        "temperature": args.temperature,
        "do_sample": bool(args.temperature and args.temperature > 0),
        "top_p": args.top_p,
        "num_beams": args.num_beams,
    }

    # Output collection
    from tqdm import tqdm as _tqdm

    start_time = time.time()
    logging.info(
        f"Starting evaluation dataset={args.dataset} size={len(ds)} model_type={args.model_type}"
    )
    print(
        f"Starting evaluation dataset={args.dataset} size={len(ds)} | Log file: {log_filename}"
    )

    predictions: List[Dict[str, Any]] = []
    for batch in _tqdm(dl, desc="Evaluating", unit="items"):
        if args.dataset in ("scanqa", "sqa3d"):
            # Prepare inputs
            question: str = batch["question"][0]
            answer: str = batch["answer"][0]
            question_id: str = batch["question_id"][0]

            video = batch["video"]
            video = [video[0].squeeze(0).to(device=model_device, dtype=model_dtype)]

            video_feature = batch["video_feat"].to(device=model_device, dtype=model_dtype)
            feat_frame_num = int(video_feature.shape[2])  # Shape: (B, C, T, H, W) -> T at index 2
            video_feat_fps = 1

            prompt_text = _build_prompt_with_image_token(
                question, getattr(model.config, "mm_use_im_start_end", False)
            )
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_vision_token(
                prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors="pt"
            ).unsqueeze(0).to(device=model_device)
            if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
                tokenizer.pad_token_id = 151643
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device=model_device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [
                KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            ]
            
            gen_local = {
                **gen_kwargs,
                "modalities": effective_modalities,
                "stopping_criteria": stopping_criteria,
            }

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_feats=video_feature,
                    video_feat_fps=torch.tensor([video_feat_fps], device=model_device),
                    feat_frame_nums=torch.tensor([feat_frame_num], device=model_device),
                    images=video,
                    image_sizes=[200],
                    attention_mask=attention_masks,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    cache_position=None,
                    **gen_local,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            predictions.append(
                {
                    "question_id": question_id,
                    "prompt": question,
                    "text": outputs,
                    "answer": answer,
                }
            )
            continue

        # AVQA, AVSD, MUSIC-AVQA paths share image+audio feature flow
        video = batch["video"]
        if isinstance(video, list) and len(video) > 0:
            video = [
                video[0].squeeze(0).to(device=model_device, dtype=model_dtype)
            ]
        else:
            # Defensive default to keep evaluation flowing
            video = [torch.zeros((32, 3, 224, 224), dtype=model_dtype, device=model_device)]

        audio_feature_raw = batch["audio_feature"]
        audio_feature, feat_frame_num = prepare_audio_feature(
            audio_feature_raw, device=model_device, dtype=model_dtype
        )

        # Build text prompt(s)
        if args.dataset == "avqa":
            vid = batch["vid"][0]
            question = batch["question"][0]
            answer = batch["answer"][0]

            prompt_text = _build_prompt_with_image_token(
                question, getattr(model.config, "mm_use_im_start_end", False)
            )
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_vision_token(
                prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors="pt"
            ).unsqueeze(0).to(device=model_device)
            if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
                tokenizer.pad_token_id = 151643
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device=model_device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [
                KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            ]
            gen_local = {
                **gen_kwargs,
                "modalities": effective_modalities,
                "stopping_criteria": stopping_criteria,
            }
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_feats=audio_feature,
                    video_feat_fps=torch.tensor([1], device=model_device),
                    feat_frame_nums=torch.tensor([feat_frame_num], device=model_device),
                    images=video,
                    image_sizes=[200],
                    attention_mask=attention_masks,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    cache_position=None,
                    **gen_local,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            predictions.append(
                {"image_id": vid, "caption": outputs, "answer": answer}
            )
            continue

        if args.dataset == "music_avqa":
            vid = batch["vid"][0]
            question = batch["question"][0]
            answer = batch["answer"][0]
            prompt_text = _build_prompt_with_image_token(
                question, getattr(model.config, "mm_use_im_start_end", False)
            )
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_vision_token(
                prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors="pt"
            ).unsqueeze(0).to(device=model_device)
            if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
                tokenizer.pad_token_id = 151643
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device=model_device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [
                KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            ]
            gen_local = {
                **gen_kwargs,
                "modalities": effective_modalities,
                "stopping_criteria": stopping_criteria,
            }
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_feats=audio_feature,
                    video_feat_fps=torch.tensor([1], device=model_device),
                    feat_frame_nums=torch.tensor([feat_frame_num], device=model_device),
                    images=video,
                    image_sizes=[200],
                    attention_mask=attention_masks,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    cache_position=None,
                    **gen_local,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            predictions.append(
                {
                    "image_id": vid,
                    "pred": outputs,
                    "question": question,
                    "answer": answer,
                }
            )
            continue

        if args.dataset == "avsd":
            vid = batch["vid"][0]
            # if vid != "HUS2X":
            #     continue
            conversation = batch["conversation"]
            if isinstance(conversation, list) and len(conversation) > 0 and isinstance(conversation[0], list):
                conversation = conversation[0]

            conv = conv_templates[args.conv_mode].copy()
            conversation_round = len(conversation) if conversation else 0
            final_answer = ""
            for i in range(conversation_round):
                q_entry = (
                    conversation[i].get("question", "")
                    if isinstance(conversation[i], dict)
                    else ""
                )
                a_entry = (
                    conversation[i].get("answer", "")
                    if isinstance(conversation[i], dict)
                    else ""
                )
                if isinstance(q_entry, list):
                    curr_q = q_entry[0]
                else:
                    curr_q = q_entry
                if isinstance(a_entry, list):
                    curr_a = a_entry[0]
                else:
                    curr_a = a_entry
                if i == 0:
                    qs = curr_q
                    if getattr(model.config, "mm_use_im_start_end", False):
                        qs = (
                            DEFAULT_IM_START_TOKEN
                            + DEFAULT_IMAGE_TOKEN
                            + DEFAULT_IM_END_TOKEN
                            + "\n"
                            + qs
                        )
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                else:
                    qs = curr_q
                conv.append_message(conv.roles[0], qs)
                if i == conversation_round - 1:
                    final_answer = curr_a
                    conv.append_message(conv.roles[1], None)
                else:
                    conv.append_message(conv.roles[1], curr_a)
            if conversation_round == 0:
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
                conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            input_ids = tokenizer_vision_token(
                prompt, tokenizer, DEFAULT_IMAGE_TOKEN, return_tensors="pt"
            ).unsqueeze(0).to(device=model_device)
            if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
                tokenizer.pad_token_id = 151643
            attention_masks = input_ids.ne(tokenizer.pad_token_id).long().to(device=model_device)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = [
                KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
            ]
            gen_local = {
                **gen_kwargs,
                "modalities": effective_modalities,
                "stopping_criteria": stopping_criteria,
            }
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    video_feats=audio_feature,
                    video_feat_fps=torch.tensor([1], device=model_device),
                    feat_frame_nums=torch.tensor([feat_frame_num], device=model_device),
                    images=video,
                    image_sizes=[200],
                    attention_mask=attention_masks,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    cache_position=None,
                    **gen_local,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()
            predictions.append(
                {"image_id": vid, "caption": outputs, "answer": final_answer}
            )
            # predictions.append(
            #     {"image_id": vid, "caption": outputs, "answer": final_answer, "question": conversation[-1]["question"]}
            # )
            continue

        raise RuntimeError("Unreachable dataset branch")

    # Save outputs as a single JSON array for compatibility with COCO evaluation tools
    pred_dir = os.path.dirname(args.pred_save) if args.pred_save else ""
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    with open(args.pred_save, "w") as f:
        json.dump(predictions, f)

    elapsed = time.time() - start_time
    logging.info(
        f"Completed {len(predictions)} items in {elapsed:.1f}s | saved: {args.pred_save}"
    )
    print(f"Saved predictions to: {args.pred_save}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Unified evaluator for UniMVU across supported datasets")
    p.add_argument("--dataset", type=str, required=True, choices=["avqa", "avsd", "music_avqa", "scanqa", "sqa3d"])
    p.add_argument(
        "--model-type",
        dest="model_type",
        type=str,
        default=BASE_MODEL_TYPE,
        help=f"Eval model family. Use '{BASE_MODEL_TYPE}' for separately trained checkpoints and '{UNIFIED_MODEL_TYPE}' for unified mixed-dataset checkpoints.",
    )
    p.add_argument("--model-path", dest="model_path", type=str, required=True)
    p.add_argument("--model-base", dest="model_base", type=str, default=None)
    p.add_argument("--model-arg-name", dest="model_arg_name", type=str, default=None)

    # Dataset paths
    p.add_argument("--annotation-file", dest="annotation_file", type=str, default=None,
                   help="For avqa/avsd/music_avqa")
    p.add_argument("--question-file", dest="question_file", type=str, default=None,
                   help="For scanqa/sqa3d")
    p.add_argument("--video-folder", type=str, required=True)
    p.add_argument("--feature-folder", type=str, required=True)
    p.add_argument("--pred-save", type=str, required=True)
    p.add_argument(
        "--data-sample-ratio",
        dest="data_sample_ratio",
        type=float,
        default=None,
        help="Fraction of the dataset to evaluate (0-1]; 0 keeps one sample; >1 uses the full dataset.",
    )

    p.add_argument("--conv-mode", type=str, default="qwen_1_5")
    p.add_argument("--num-workers", dest="num_workers", type=int, default=2)
    p.add_argument("--for_get_frames_num", type=int, default=32)
    p.add_argument("--feat-type", dest="feat_type", type=str, default="gama",
                   help="music_avqa feature type: languagebind|imagebind|gama")

    # Generation
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_beams", type=int, default=1)

    # Model config pass-through
    p.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    p.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    p.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    p.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    p.add_argument("--mm_newline_position", type=str, default="no_token")
    p.add_argument("--modalities", type=str, nargs='+',
                   help="Modalities to use (e.g., --modalities video audio)")
    return p


if __name__ == "__main__":
    parser = build_parser()
    ns = parser.parse_args()
    eval_args = EvalArgs(
        dataset=ns.dataset,
        model_type=ns.model_type,
        model_path=ns.model_path,
        model_base=ns.model_base,
        model_arg_name=ns.model_arg_name,
        annotation_file=ns.annotation_file,
        question_file=ns.question_file,
        video_folder=ns.video_folder,
        feature_folder=ns.feature_folder,
        pred_save=ns.pred_save,
        conv_mode=ns.conv_mode,
        num_workers=ns.num_workers,
        for_get_frames_num=ns.for_get_frames_num,
        data_sample_ratio=ns.data_sample_ratio,
        feat_type=ns.feat_type,
        temperature=ns.temperature,
        top_p=ns.top_p,
        num_beams=ns.num_beams,
        mm_resampler_type=ns.mm_resampler_type,
        mm_spatial_pool_stride=ns.mm_spatial_pool_stride,
        mm_spatial_pool_out_channels=ns.mm_spatial_pool_out_channels,
        mm_spatial_pool_mode=ns.mm_spatial_pool_mode,
        mm_newline_position=ns.mm_newline_position,
        modalities=ns.modalities,
    )
    evaluate(eval_args)
