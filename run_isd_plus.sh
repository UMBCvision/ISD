#!/usr/bin/env bash

set -x
set -e

CUDA_VISIBLE_DEVICES=$1 python train_isd_plus.py \
    --num_workers 24 \
    --print_freq 100 \
    --momentum 0.99 \
    --queue_size 128000 \
    --temp_t 0.01 \
    --temp_s 0.1 \
    --learning_rate 0.05 \
    --epochs 200  \
    --cos \
    --arch resnet50 \
    --augmentation 'weak/weak' \
    --checkpoint_path isd_plus_1_aug_ww_resnet50 \
    /nfs/ada/hpirsiav/datasets/imagenet

CUDA_VISIBLE_DEVICES=$1 python train_isd_plus.py \
    --num_workers 24 \
    --print_freq 100 \
    --momentum 0.99 \
    --queue_size 128000 \
    --temp_t 0.01 \
    --temp_s 0.1 \
    --learning_rate 0.05 \
    --epochs 200  \
    --cos \
    --arch resnet50 \
    --augmentation 'strong/weak' \
    --checkpoint_path isd_plus_2_aug_sw_resnet50 \
    /nfs/ada/hpirsiav/datasets/imagenet
