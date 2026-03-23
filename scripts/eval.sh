#!/bin/bash
set -e
set -o pipefail

# Switch between --annotation-file and --question-file depending on the dataset.
CUDA_VISIBLE_DEVICES=0 python unified_eval.py \
    --dataset avqa \
    --model-path /path/to/checkpoint \
    --model-base lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --model-arg-name VideoFeatModelArgumentsUniMVU \
    --model-type unimvu \
    --annotation-file /path/to/val.json \
    --video-folder /path/to/videos \
    --feature-folder /path/to/features \
    --pred-save ./eval_output/unimvu_eval.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen
