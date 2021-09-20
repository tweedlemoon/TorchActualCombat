import torch
import torchvision
import numpy as np
import sys
import os

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# 读数据
directory = os.path.abspath(os.path.join(os.getcwd(), "../", "Datasets"))
if not os.path.exists(directory):
    os.makedirs(directory)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=None, root=directory)
