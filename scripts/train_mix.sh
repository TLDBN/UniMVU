#!/bin/bash
set -e
set -o pipefail

# Edit the dataset lists and output path before launching.
NCCL_P2P_DISABLE=1 \
deepspeed --master_port 60000 train_uni.py \
    --deepspeed ./scripts/zero2_flops_uni_7B.json \
    --lora_enable True \
    --lora_alpha 128 \
    --data_class VideoFeatMixedDataArguments \
    --datasets dataset_a dataset_b \
    --annotation_path \
        /path/to/dataset_a_train.json \
        /path/to/dataset_b_train.json \
    --fast_path_mapping_path \
        /path/to/dataset_a_fast_mapping.json \
        /path/to/dataset_b_fast_mapping.json \
    --slow_path_mapping_path \
        /path/to/dataset_a_video_mapping.json \
        /path/to/dataset_b_video_mapping.json \
    --data_root \
        /path/to/dataset_a_fast_features \
        /path/to/dataset_b_fast_features \
    --slow_path_data_root \
        /path/to/dataset_a_raw_videos \
        /path/to/dataset_b_raw_videos \
    --use_fast_feat True \
    --use_slow True \
    --shuffle True \
    --mix_sampling_alpha 0.5 \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --version conv_llava_ov_qwen \
    --model_class VideoFeatModelArgumentsUniMVUUni_7B \
    --model_type unimvu_uni \
    --output_dir /path/to/checkpoints/unimvu_mix \
    --extra_trainable_modules modality_aggregator modality_projectors modality_special_token_aggregator modality_tokens \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --ddp_find_unused_parameters True \
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
    --feat_combine_method add \
    --num_cross_modality_hidden_layers 1 \
    --support_modalities video audio 3d_feature dense_video \
    --modality_input_dims video:1024,audio:1024,3d_feature:1024,dense_video:1024
