#!/usr/bin/env bash

set -x
set -e

CUDA_VISIBLE_DEVICES=$1 python train_byol.py \
  --save_freq 5 \
  --print_freq 100 \
  --num_workers 12 \
  --cos \
  --learning_rate 0.05 \
  --epochs 200 \
  --momentum 0.99 \
  --arch resnet18 \
  --checkpoint_path output/byol_1_full_imagenet_augv2_m_0x99_cos_lr_05_epochs_200_resnet18 \
  /datasets/imagenet

