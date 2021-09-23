# %%
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import os

sys.path.append("..")
import d2lzh_pytorch as d2l

# %%
# 读数据
directory = os.path.abspath(os.path.join(os.getcwd(), "../", "Datasets"))
if not os.path.exists(directory):
    os.makedirs(directory)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=None, root=directory)

# %%
# 建模，此处连net都不需要定义，直接拿nn里的Linear
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

# %%
# 训练模型
# loss
loss = torch.nn.CrossEntropyLoss()
# 最优化方法
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
