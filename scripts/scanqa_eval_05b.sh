#!/bin/bash
set -e
set -o pipefail

mkdir -p /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction

CUDA_VISIBLE_DEVICES=7 python unified_eval.py \
    --dataset scanqa \
    --model-path /share_1/users/bonan_ding/PAVE_ckpt/checkpoints/unimvuv3_scanqa_0.5B_4epoch_bs16_lora_a128_2e-5 \
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
