#!/bin/bash
set -e
set -o pipefail

# Edit the dataset paths and optimization settings before launching.
NCCL_P2P_DISABLE=1 \
deepspeed --master_port 60000 train.py \
    --deepspeed ./scripts/zero2_flops_uni_05B.json \
    --lora_enable True \
    --lora_alpha 128 \
    --data_class VideoFeatMixedDataArguments \
    --annotation_path /path/to/train.json \
    --fast_path_mapping_path /path/to/fast_feature_mapping.json \
    --slow_path_mapping_path /path/to/video_mapping.json \
    --data_root /path/to/fast_features \
    --slow_path_data_root /path/to/raw_videos \
    --use_fast_feat True \
    --use_slow True \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --version conv_llava_ov_qwen \
    --model_class VideoFeatModelArgumentsUniMVU \
    --model_type unimvu \
    --output_dir /path/to/checkpoints/unimvu_single \
    --extra_trainable_modules modality_aggregator input_mapping modality_special_token_aggregator modality_tokens \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --bf16 True \
    --tf32 True \
    --mm_newline_position grid \
    --mm_spatial_pool_mode bilinear \
    --feat_combine_method concat \
    --fast_feat_type audio \
    --num_cross_modality_hidden_layers 1 \
    --support_modalities video audio
