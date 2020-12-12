import argparse
import re
import dill
import torch
import numpy as np
from models.resnet_byol import resnet50


def main():
    parser = argparse.ArgumentParser(description='Convert BYOL weights from JAX/Haiku to PyTorch')
    parser.add_argument('byol_wts_path', default='pretrain_res50x1.pkl', type=str,
            help='path to the BYOL weights file to convert')
    parser.add_argument('pytorch_wts_path', default='pretrain_res50x1.pth.tar', type=str,
            help='optional path to the output PyTorch weights file name')

    args = parser.parse_args()

    ckpt = load_checkpoint(args.byol_wts_path)
    online_weights = ckpt['experiment_state'].online_params
    online_bn_states = ckpt['experiment_state'].online_state
    target_weights = ckpt['experiment_state'].target_params
    target_bn_states = ckpt['experiment_state'].target_state
    encoder_q = convert_weights(online_weights, online_bn_states)
    encoder_k = convert_weights(target_weights, target_bn_states)
    predict_q = {k: v for k, v in encoder_q.items() if 'predict_q' in k}
    encoder_q = {k: v for k, v in encoder_q.items() if 'predict_q' not in k}
    encoder_k = {k: v for k, v in encoder_k.items() if 'predict_q' not in k}
    encoder_q = {'encoder_q.{}'.format(k): v for k, v in encoder_q.items()}
    encoder_k = {'encoder_k.{}'.format(k): v for k, v in encoder_k.items()}
    state_dict = {}
    state_dict.update(predict_q)
    state_dict.update(encoder_q)
    state_dict.update(encoder_k)

    torch.save(state_dict, args.pytorch_wts_path)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint_data = dill.load(checkpoint_file)
        print('=> loading checkpoint from {}, saved at step {}'.format(
            checkpoint_path, checkpoint_data['step']
        ))
        return checkpoint_data


def convert_weights(weights, bn_states):
    print('==> process weights')
    state_dict = {}
    for k, v in zip(weights.keys(), weights.values()):
        if 'classifier' in k:
            continue
        f_k = k
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9])/~/(conv|batchnorm)_([0-9])',
            lambda m: 'layer{}.{}.{}{}'.format(int(m[1])+1, int(m[2]), m[3], int(m[4])+1),
            f_k
        )
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9])/~/shortcut_(conv|batchnorm)',
            lambda m: 'layer{}.{}.{}'.format(int(m[1])+1, int(m[2]), 'downsample.' + m[3])\
                .replace('conv', '0').replace('batchnorm', '1'),
            f_k
        )
        f_k = re.sub(
            '.*initial_(conv|batchnorm)(_1)?',
            lambda m: '{}'.format(m[1] + '1'),
            f_k
        )
        f_k = f_k.replace('batchnorm', 'bn')

        f_k = f_k.replace('projector/linear_1', 'fc.3')
        f_k = f_k.replace('projector/linear', 'fc.0')
        f_k = f_k.replace('projector/batch_norm', 'fc.1')

        f_k = f_k.replace('predictor/linear_1', 'predict_q.3')
        f_k = f_k.replace('predictor/linear', 'predict_q.0')
        f_k = f_k.replace('predictor/batch_norm', 'predict_q.1')

        for p_k, p_v in zip(v.keys(), v.values()):
            p_k = p_k.replace('w', '.weight')
            p_k = p_k.replace('b', '.bias')
            p_k = p_k.replace('offset', '.bias')
            p_k = p_k.replace('scale', '.weight')
            ff_k = f_k + p_k
            p_v = torch.from_numpy(p_v)
            print(k, ff_k, p_v.shape)
            if 'conv' in ff_k or 'downsample.0' in ff_k:
                state_dict[ff_k] = p_v.permute(3, 2, 0, 1)
            elif 'bn' in ff_k or 'downsample.1' in ff_k or 'fc.1' in ff_k or 'predict_q.1' in ff_k:
                state_dict[ff_k] = p_v.squeeze()
            elif ('fc.' in ff_k or 'predict_q.' in ff_k) and '.weight' in ff_k:
                state_dict[ff_k] = p_v.permute(1, 0)
            else:
                state_dict[ff_k] = p_v

    print('==> process bn_states')
    for k, v in zip(bn_states.keys(), bn_states.values()):
        if 'classifier' in k:
            continue
        f_k = k
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9])/~/(conv|batchnorm)_([0-9])',
            lambda m: 'layer{}.{}.{}{}'.format(int(m[1])+1, int(m[2]), m[3], int(m[4])+1),
            f_k
        )
        f_k = re.sub(
            '.*block_group_([0-9]).*block_([0-9])/~/shortcut_(conv|batchnorm)',
            lambda m: 'layer{}.{}.{}'.format(int(m[1])+1, int(m[2]), 'downsample.' + m[3])\
                .replace('conv', '0').replace('batchnorm', '1'),
            f_k
        )
        f_k = re.sub(
            '.*initial_(conv|batchnorm)',
            lambda m: '{}'.format(m[1] + '1'),
            f_k
        )
        f_k = f_k.replace('batchnorm', 'bn')
        f_k = f_k.replace('projector/linear_1', 'fc.3')
        f_k = f_k.replace('projector/linear', 'fc.0')
        f_k = f_k.replace('projector/batch_norm', 'fc.1')

        f_k = f_k.replace('predictor/linear_1', 'predict_q.3')
        f_k = f_k.replace('predictor/linear', 'predict_q.0')
        f_k = f_k.replace('predictor/batch_norm', 'predict_q.1')
        f_k = f_k.replace('/~/mean_ema', '.running_mean')
        f_k = f_k.replace('/~/var_ema', '.running_var')
        assert np.abs(v['average'] - v['hidden']).sum() == 0
        print(k, f_k, v['average'].shape)
        state_dict[f_k] = torch.from_numpy(v['average']).squeeze()
    return state_dict


if __name__ == '__main__':
    main()

