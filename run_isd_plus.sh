#!/usr/bin/env bash

#SBATCH --account=pi_hpirsiav
#SBATCH --time=9-00:00:00
#SBATCH --job-name=isd_6_plus_tt_0x02_ts_0x20_cos_lr_0x01_m_0x99_aug_ws_mlp_resnet18
#SBATCH --output=logs/isd_6_plus_tt_0x02_ts_0x20_cos_lr_0x01_m_0x99_aug_ws_mlp_resnet18.txt
#SBATCH --error=logs/isd_6_plus_tt_0x02_ts_0x20_cos_lr_0x01_m_0x99_aug_ws_mlp_resnet18.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G

set -x
set -e

python train_isd_plus.py \
    --momentum 0.99 \
    --temp_t 0.02 \
    --temp_s 0.20 \
    --learning_rate 0.01 \
    --cos \
    --arch resnet18 \
    --augmentation 'weak/strong' \
    --checkpoint_path output/isd_6_plus_tt_0x02_ts_0x20_cos_lr_0x01_m_0x99_aug_ws_mlp_resnet18 \
    /nfs/ada/hpirsiav/datasets/imagenet

# python train_isd_plus.py \
#     --momentum 0.99 \
#     --temp_t 0.02 \
#     --temp_s 0.02 \
#     --learning_rate 0.01 \
#     --cos \
#     --arch resnet18 \
#     --augmentation 'weak/strong' \
#     --checkpoint_path output/isd_5_plus_cos_lr_0x01_m_0x99_aug_ws_mlp_resnet18 \
#     /nfs/ada/hpirsiav/datasets/imagenet

# CUDA_VISIBLE_DEVICES=$1 python train_isd_plus.py \
#     --num_workers 24 \
#     --print_freq 100 \
#     --momentum 0.99 \
#     --queue_size 128000 \
#     --temp_t 0.01 \
#     --temp_s 0.1 \
#     --learning_rate 0.05 \
#     --epochs 200  \
#     --cos \
#     --arch resnet50 \
#     --augmentation 'weak/weak' \
#     --checkpoint_path isd_plus_1_aug_ww_resnet50 \
#     /nfs/ada/hpirsiav/datasets/imagenet

# CUDA_VISIBLE_DEVICES=$1 python train_isd_plus.py \
#     --num_workers 24 \
#     --print_freq 100 \
#     --momentum 0.99 \
#     --queue_size 128000 \
#     --temp_t 0.01 \
#     --temp_s 0.1 \
#     --learning_rate 0.05 \
#     --epochs 200  \
#     --cos \
#     --arch resnet50 \
#     --augmentation 'strong/weak' \
#     --checkpoint_path isd_plus_2_aug_sw_resnet50 \
#     /nfs/ada/hpirsiav/datasets/imagenet
