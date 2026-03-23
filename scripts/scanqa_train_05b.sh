#!/bin/bash
set -e
set -o pipefail

mkdir -p /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction

WANDB__SERVICE_WAIT=500 \
deepspeed --master_port 60000 train.py \
    --deepspeed ./scripts/zero2_flops_uni_05B.json \
    --lora_enable True \
    --lora_alpha 128 \
    --data_class VideoFeatMixedDataArguments \
    --annotation_path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/scanqa_train_instruct.json \
    --fast_path_mapping_path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/scannet_from_scan_id_to_video_feature.json \
    --slow_path_mapping_path /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/scannet_from_scan_id_to_video_folder.json \
    --data_root /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/video_features_new \
    --slow_path_data_root /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/posed_images_new \
    --use_fast_feat True \
    --use_slow True \
    --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --version conv_llava_ov_qwen \
    --model_class VideoFeatModelArgumentsUniMVU \
    --model_type unimvu \
    --output_dir ./checkpoints/unimvu_scanqa_05b \
    --extra_trainable_modules modality_aggregator input_mapping modality_special_token_aggregator modality_tokens \
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
    --feat_combine_method add \
    --fast_feat_type 3d_feature \
    --num_cross_modality_hidden_layers 1 \
    --support_modalities video 3d_feature

CUDA_VISIBLE_DEVICES=0 python unified_eval.py \
    --dataset scanqa \
    --model-path ./checkpoints/unimvu_scanqa_05b \
    --model-base lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --model-arg-name VideoFeatModelArgumentsUniMVU \
    --model-type unimvu \
    --question-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/llava-3d-scanqa_val_question.json \
    --video-folder /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/posed_images_new \
    --feature-folder /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/video_features_new \
    --pred-save /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction/unimvu_scanqa_05b.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen

python tools/3d/scanqa/scanqa_evaluator.py \
    --gt-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/scannet/llava3d_scanqa_val_answer.json \
    --results-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction/unimvu_scanqa_05b.json
