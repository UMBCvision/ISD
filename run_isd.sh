#!/usr/bin/env bash

set -x
set -e

# CUDA_VISIBLE_DEVICES=$1 python train_isd_r18.py \
#     --save_freq 5 \
#     --num_workers 16 \
#     --epochs 200 \
#     --learning_rate 0.01 \
#     --lr_decay_epochs '140,180' \
#     --momentum 0.999 \
#     --arch resnet18 \
#     --augmentation 'weak/strong' \
#     --checkpoint_path output/isd_1_step_lr_0x01_140_180_m_0x999_aug_ws_resnet18 \
#     /nfs/ada/hpirsiav/datasets/imagenet

# CUDA_VISIBLE_DEVICES=$1 python train_isd_r18.py \
#     --save_freq 5 \
#     --num_workers 16 \
#     --epochs 200 \
#     --learning_rate 0.01 \
#     --cos \
#     --momentum 0.999 \
#     --arch resnet18 \
#     --augmentation 'strong/strong' \
#     --checkpoint_path output/isd_2_cos_lr_0x01_m_0x999_aug_ss_resnet18 \
#     /nfs/ada/hpirsiav/datasets/imagenet

# CUDA_VISIBLE_DEVICES=$1 python train_isd_r18.py \
#     --save_freq 5 \
#     --num_workers 16 \
#     --epochs 200 \
#     --learning_rate 0.01 \
#     --lr_decay_epochs '140,180' \
#     --momentum 0.99 \
#     --arch resnet18 \
#     --augmentation 'strong/strong' \
#     --checkpoint_path output/isd_3_step_lr_0x01_140_180_m_0x99_aug_ss_resnet18 \
#     /nfs/ada/hpirsiav/datasets/imagenet

CUDA_VISIBLE_DEVICES=$1 python train_isd_r18.py \
    --save_freq 5 \
    --num_workers 16 \
    --epochs 200 \
    --learning_rate 0.01 \
    --cos \
    --momentum 0.99 \
    --arch resnet18 \
    --augmentation 'weak/strong' \
    --checkpoint_path output/isd_4_cos_lr_0x01_m_0x99_aug_ws_resnet18 \
    /nfs/ada/hpirsiav/datasets/imagenet

# CUDA_VISIBLE_DEVICES=$1 python train_isd.py \
#   --save_freq 5 \
#   --num_workers 16 \
#   --print_freq 100 \
#   --queue_size 128000 \
#   --temp 0.020 \
#   --epochs 200 \
#   --lr_decay_epochs '140,180' \
#   --arch resnet18 \
#   --checkpoint_path output/isd_1_t_020_augv2_lr_decay_140_180_epochs_200_resnet18 \
#   /datasets/imagenet

# CUDA_VISIBLE_DEVICES=$1 python train_isd.py \
#   --save_freq 1 \
#   --num_workers 16 \
#   --print_freq 100 \
#   --queue_size 1280 \
#   --learning_rate 0.005 \
#   --temp 0.020 \
#   --epochs 10 \
#   --lr_decay_epochs '7,9' \
#   --arch resnet50 \
#   --weights weights/byol_resnet50_everything.pth.tar \
#   --checkpoint_path output/isd_2_init_byol_everything_lr_005_step_7_9_epochs_10_k12h_resnet50 \
#   /datasets/imagenet

