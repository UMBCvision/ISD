#!/usr/bin/env bash

#SBATCH --account=pi_hpirsiav
#SBATCH --time=20:00:00
#SBATCH --job-name=knn_isd_3
#SBATCH --output=logs/knn_isd_3.txt
#SBATCH --error=logs/knn_isd_3.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G

set -x
set -e

for exp_dir in output/isd_3_*_resnet18
do
    python eval_knn.py \
        -j 16 \
        -b 256 \
        --arch resnet18 \
        --weights $exp_dir/ckpt_epoch_200.pth \
        --save $exp_dir \
        /nfs/ada/hpirsiav/datasets/imagenet
done


