# %%
import torch
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
# 定义参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(True)


# %%
# 激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# %%
# 定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# %%
# 定义损失函数
# 直接调用交叉熵
loss = torch.nn.CrossEntropyLoss()

# %%
# 训练模型

num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
