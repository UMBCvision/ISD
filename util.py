from __future__ import print_function
import math

import torch
import numpy as np


# NOTE: assumes that the epoch starts with 1
def adjust_learning_rate(epoch, opt, optimizer):
    if hasattr(opt, 'cos') and opt.cos:
        # NOTE: since epoch starts with 1, we have to subtract 1
        new_lr = opt.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch-1) / opt.epochs))
        print('LR: {}'.format(new_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
        if steps > 0:
            new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
            print('LR: {}'.format(new_lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    meter = AverageMeter()
