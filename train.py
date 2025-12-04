import argparse
import torch
import os
from utils.regression_trainer import Reg_Trainer


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', default="train", type=str,
                        help='what is it?')
    parser.add_argument('--seed', default=314, type=int, help='if not using seed, please set as -1')
    parser.add_argument('--crop-size', default=384, type=int,
                        help='the cropped size of the training data')
    parser.add_argument('--concat-size', default=224, type=int,
                        help='the concat size of the training data')
    parser.add_argument('--downsample-ratio', default=8, type=int,
                        help='the downsample ratio of the model')
    parser.add_argument('--data-dir', default='data/FSC',
                        help='the directory of the data')
    parser.add_argument('--save-dir', default='history',
                        help='the directory for saving models and training logs')

    parser.add_argument('--max-num', default=2, type=int,
                        help='the maximum number of saved models ')
    parser.add_argument('--resume', default="/home/ubuntu/jyt/checkpoints/mae/mae_pretrain_vit_base_full.pth",
                        help='the path of the resume training model')
    parser.add_argument('--batch-size', default=4, type=int,
                        help='the number of samples in a batch')
    parser.add_argument('--stride', default=384, type=int,
                        help='the stride for patchify')
    parser.add_argument('--beta', default=1e-4, type=float,
                        help='the initialization value of beta')

    # Optimizer
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='the learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='the number of workers')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--start-epoch', default=0, type=int,
                        help='the number of starting epoch')
    parser.add_argument('--epochs', default=200, type=int,
                        help='the maximum number of training epoch')
    parser.add_argument('--start-val', default=10, type=int,
                        help='the starting epoch for validation')
    parser.add_argument('--val-epoch', default=10, type=int,
                        help='the number of epoch between validation')
    parser.add_argument('--time-step', default=0, type=int,
                        help='the number of epoch between validation')
    parser.add_argument("--scale", default=60, type=int, help="scale size")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    torch.backends.cudnn.benchmark = True
    trainer = Reg_Trainer(args)
    trainer.setup()
    trainer.train()