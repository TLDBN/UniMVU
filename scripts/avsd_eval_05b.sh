#!/bin/bash
set -e
set -o pipefail

mkdir -p /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction

CUDA_VISIBLE_DEVICES=0 python unified_eval.py \
    --dataset avsd \
    --model-path ./checkpoints/unimvu_avsd_05b \
    --model-base lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --model-arg-name VideoFeatModelArgumentsUniMVU \
    --model-type unimvu \
    --annotation-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json \
    --video-folder /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/avsd/Charades_vu17_test_480 \
    --feature-folder /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat \
    --pred-save /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction/unimvu_avsd_05b.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen

python tools/audio/avsd/run_coco_eval.py \
    --gt-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/avsd/coco_version_test_gt.json \
    --results-file /share_1/users/bonan_ding/PAVE_data/data/video_instruction_tuning/prediction/unimvu_avsd_05b.json
