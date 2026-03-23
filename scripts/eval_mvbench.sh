#!/bin/bash
set -e
set -o pipefail

export UNIMVU_MVBENCH_VIDEO_ROOT=/share_1/users/bonan_ding/PAVE_data/MVBench/
export UNIMVU_MVBENCH_FEATURE_ROOT=/share_1/users/bonan_ding/PAVE_data/MVBench/languagebind_feat

CUDA_VISIBLE_DEVICES=6 python lmms_eval_start.py \
    --model unimvu_uni \
    --tasks mvbench \
    --model_args model_path=/share_1/users/bonan_ding/PAVE_ckpt/checkpoints/unimvuv3_uni_7B_r64_a128_all_mix_alpha05_epoch2_bs64_lr2e-5/checkpoint-4000,model_base=lmms-lab/llava-onevision-qwen2-7b-ov,model_arg_name=VideoFeatModelArgumentsUniMVUUni_7B,conv_template=conv_llava_ov_qwen,fast_feat_type=dense_video,slow_feat_type=raw_video \
    --output_path ./logs/unimvu/mvbench/unimvuv3_uni_7B_r64_a128_all_mix_alpha05_epoch2_bs64_lr2e-5/checkpoint-4000 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix unimvu_uni
