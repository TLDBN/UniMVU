"""UniMVU training utilities."""

from __future__ import annotations

import torch
import transformers
from transformers import AutoConfig, BitsAndBytesConfig

from libs import conversation_lib as conversation_lib
from libs.dataset.image_dataset import smart_tokenizer_and_embedding_resize
from libs.model.multimodal_LMM._unimvu_base import (
    UniMVUQwen2Config,
    UniMVUQwen2ForCausalLM,
)
from libs.model.multimodal_LMM._unimvu_mix import (
    UniMVUUnifiedConfig,
    UniMVUUnifiedForCausalLM,
)
from libs.utils.train_utils import rank0_print


MODEL_TYPE_MAPPING = {
    "unimvu": (UniMVUQwen2ForCausalLM, UniMVUQwen2Config),
    "unimvu_uni": (UniMVUUnifiedForCausalLM, UniMVUUnifiedConfig),
}


def _get_lora_target_modules(model):
    layers = None
    if hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
        layers = model.base_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "get_model") and hasattr(model.get_model(), "layers"):
        layers = model.get_model().layers
    if layers is None:
        raise AttributeError("Unable to locate transformer layers for LoRA target selection.")

    return [f"model.layers.{name}" for name, _ in layers.named_modules() if "proj" in name]


def prepare_video_model_v2(training_args, model_args, data_args, compute_dtype, attn_implementation):
    """Build the released UniMVU model and tokenizer for training."""

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector", "temporal_aggregator"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    model_type = str(getattr(model_args, "model_type", "unimvu")).lower()
    if model_type not in MODEL_TYPE_MAPPING:
        raise NotImplementedError(
            f"Model type '{model_type}' is not supported. Available types: {list(MODEL_TYPE_MAPPING.keys())}"
        )

    model_class, config_class = MODEL_TYPE_MAPPING[model_type]
    base_llm_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config = config_class(model_config=model_args, **base_llm_config.to_dict())
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        config=config,
        **bnb_model_from_pretrained_args,
    )
    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.model_max_length = training_args.model_max_length

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version.startswith("Qwen2Tokenizer"):
        tokenizer.pad_token = "<|endoftext|>"
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if hasattr(model, "initialize_vision_tokenizer"):
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        selected_module = _get_lora_target_modules(model)
        patterns = getattr(training_args, "extra_trainable_modules", None)
        if patterns:
            if not isinstance(patterns, (list, tuple)):
                patterns = [patterns]
            selected_module = [module for module in selected_module if not any(p in module for p in patterns)]

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=selected_module,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

        if patterns:
            enabled_params = []
            for param_name, param in model.named_parameters():
                if any(pattern in param_name for pattern in patterns):
                    param.requires_grad = True
                    enabled_params.append(param_name)
            if enabled_params:
                rank0_print(f"Enabled full-training parameters: {len(enabled_params)}")
    elif getattr(training_args, "extra_trainable_modules", None):
        patterns = training_args.extra_trainable_modules
        if not isinstance(patterns, (list, tuple)):
            patterns = [patterns]
        model.requires_grad_(False)
        for param_name, param in model.named_parameters():
            if any(pattern in param_name for pattern in patterns):
                param.requires_grad = True

    if model_args.video_tower is not None:
        model_args.image_size = data_args.image_size
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        video_tower = model.get_video_tower()
        if video_tower is not None:
            video_tower.to(
                dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                device=training_args.device,
            )
            video_tower.data_type = torch.bfloat16 if training_args.bf16 else torch.float16

        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.tune_temporal_aggregator = (
            training_args.tune_temporal_aggregator
        ) = model_args.tune_temporal_aggregator
        if model_args.tune_temporal_aggregator:
            model.requires_grad_(False)
            for parameter in model.get_model().temporal_aggregator.parameters():
                parameter.requires_grad = True
        if training_args.bits in [4, 8]:
            model.get_model().temporal_aggregator.to(
                dtype=compute_dtype, device=training_args.device
            )

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and training_args.bf16:
                module.to(torch.bfloat16)
            if "norm" in name:
                module.to(torch.float32)
            if ("lm_head" in name or "embed_tokens" in name) and hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module.to(torch.bfloat16)

    total_params = sum(
        parameter.ds_numel if hasattr(parameter, "ds_numel") else parameter.numel()
        for parameter in model.parameters()
    )
    trainable_params = sum(
        parameter.ds_numel if hasattr(parameter, "ds_numel") else parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Total parameters: ~{total_params / 1e6:.2f} MB)")
    print(f"Trainable parameters: ~{trainable_params / 1e6:.2f} MB)")

    return model, tokenizer
