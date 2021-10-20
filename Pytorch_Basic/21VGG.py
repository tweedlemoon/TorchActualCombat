import time
import torch
from torch import nn, optim

import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 这里会使宽高减半
    # 类似于*args,**kwargs，输入如果是一个字典就会存在**kwargs里，如果是数据就会存在*args，*args本质是tuple，**kwargs的本质是dict
    return nn.Sequential(*blk)
