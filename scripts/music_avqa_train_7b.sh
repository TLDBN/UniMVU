#!/bin/bash
set -e
set -o pipefail

mkdir -p /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction

WANDB__SERVICE_WAIT=500 \
deepspeed --master_port 60000 train.py \
    --deepspeed ./scripts/zero2_flops_uni_7B.json \
    --lora_enable True \
    --lora_alpha 128 \
    --data_class VideoFeatMixedDataArguments \
    --annotation_path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa/music_avqa_train_instruct_duplicate_audio.json \
    --fast_path_mapping_path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa/music_avqa_all_imagebind_feature_mapping.json \
    --slow_path_mapping_path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa/music_avqa_all_videos_mapping.json \
    --data_root /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa \
    --slow_path_data_root /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa \
    --use_fast_feat True \
    --use_slow True \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-7b-ov \
    --version conv_llava_ov_qwen \
    --model_class VideoFeatModelArgumentsUniMVU_7B \
    --model_type unimvu \
    --output_dir ./checkpoints/unimvu_music_avqa_7b \
    --extra_trainable_modules modality_aggregator modality_projectors modality_special_token_aggregator modality_tokens \
    --input_dim 1024 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
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

CUDA_VISIBLE_DEVICES=0 python unified_eval.py \
    --dataset music_avqa \
    --model-path ./checkpoints/unimvu_music_avqa_7b \
    --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
    --model-arg-name VideoFeatModelArgumentsUniMVU_7B \
    --model-type unimvu \
    --annotation-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa/updated_avqa-test.json \
    --video-folder /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa \
    --feature-folder /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa \
    --pred-save /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction/unimvu_music_avqa_7b.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen \
    --feat-type imagebind

python tools/audio/music-avqa/calculate_acc.py \
    --prediction-path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction/unimvu_music_avqa_7b.json \
    --annotation-path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/music_avqa/updated_avqa-test.json
