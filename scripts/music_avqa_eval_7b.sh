#!/bin/bash
set -e
set -o pipefail

mkdir -p /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction

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
