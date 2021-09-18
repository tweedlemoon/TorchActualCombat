# %%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch

import d2lzh_pytorch as d2l

# %%

directory = os.path.abspath(os.path.join(os.getcwd(), "../", "Datasets"))
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.abspath(os.path.join(os.getcwd(), "../Datasets/", "FashionMNIST"))
if not os.path.exists(directory):
    os.makedirs(directory)

mnist_train = torchvision.datasets.FashionMNIST(root=directory, train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=directory, train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

print('==================================')

# 可以通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# %%
