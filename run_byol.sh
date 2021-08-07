#!/usr/bin/env bash

#SBATCH --account=pi_hpirsiav
#SBATCH --time=20-00:00:00
#SBATCH --job-name=byol_1_aug_ww_m_0x99_cos_lr_05_ep_200_resnet50
#SBATCH --output=logs/byol_1_aug_ww_m_0x99_cos_lr_05_ep_200_resnet50.txt
#SBATCH --error=logs/byol_1_aug_ww_m_0x99_cos_lr_05_ep_200_resnet50.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G

set -x
set -e

python train_byol_aug.py \
    --save_freq 10 \
    --print_freq 100 \
    --num_workers 24 \
    --cos \
    --learning_rate 0.05 \
    --epochs 200 \
    --momentum 0.99 \
    --augmentation 'weak/weak' \
    --arch resnet50 \
    --checkpoint_path output/byol_1_aug_ww_m_0x99_cos_lr_05_ep_200_resnet50 \
    /nfs/ada/hpirsiav/datasets/imagenet

