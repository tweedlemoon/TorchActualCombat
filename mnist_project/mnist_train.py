'''
mnist_project
使用三层神经网络解决MNIST问题
课程跟随练习代码
'''

# torch常用导入
import torch
from torch import nn
from torch.nn import functional as F
# torch相关工具包
from torch import optim
import torchvision

from matplotlib import pyplot as plt

# 包
from mnist_project.utils import *

'''
1. 加载数据集
70k图片
60ktraining
10ktest
'''
# batch_size为一次处理图片的数量，充分利用cpu能力，从而降低处理时间

batch_size = 64
train_load = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data',
    train=True,
    download=True,
    # 下载下来的是numpy格式，先ToTensor
    # 然后使用正则化，让数据集尽量均匀分布在0附近，不正则效果会差点
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size,
    shuffle=True)

test_load = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data/',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size,
    shuffle=False)

# 显示加载图片
x, y = next(iter(train_load))
print(x.shape, y.shape)
plot_image(x, y, 'sample')
