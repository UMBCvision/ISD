# ISD

This is the code for the paper ["ISD: Self-Supervised Learning by Iterative Similarity Distillation"](https://www.csee.umbc.edu/~hpirsiav/papers/ISD_iccv21.pdf)

```
@InProceedings{Tejankar_2021_ICCV,
    author    = {Tejankar, Ajinkya and Koohpayegani, Soroush Abbasi and Pillai, Vipin and Favaro, Paolo and Pirsiavash, Hamed},
    title     = {ISD: Self-Supervised Learning by Iterative Similarity Distillation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9609-9618}
}

@misc{tejankar2020isd,
      title={ISD: Self-Supervised Learning by Iterative Similarity Distillation}, 
      author={Ajinkya Tejankar and Soroush Abbasi Koohpayegani and Vipin Pillai and Paolo Favaro and Hamed Pirsiavash},
      year={2020},
      eprint={2012.09259},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Requirements

- Python >= 3.7.6
- PyTorch >= 1.4
- torchvision >= 0.5.0
- faiss-gpu >= 1.6.1

# Training

Following command can be used to train the ISD ResNet18 method

```
CUDA_VISIBLE_DEVICES=0,1 python train_isd.py \
  --save_freq 5 \
  --num_workers 16 \
  --print_freq 100 \
  --queue_size 128000 \
  --temp 0.020 \
  --epochs 200 \
  --lr_decay_epochs '140,180' \
  --arch resnet18 \
  --checkpoint_path output/isd_1_t_020_augv2_lr_decay_140_180_epochs_200_resnet18 \
  /datasets/imagenet
```

To train ISD ResNet50 use following command: 
```
python train_isd_plus.py \
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
  --weak_strong \
  --checkpoint_path <CHECKPOINT PATH> \
  <DATASET PATH>
```

Following command can be used to train the BYOL method. This version of BYOL works with small batchces (256) and SGD optimizer. 

```
CUDA_VISIBLE_DEVICES=0,1 python train_byol.py \
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
  ```
  
  
## License

This project is under the MIT license.
