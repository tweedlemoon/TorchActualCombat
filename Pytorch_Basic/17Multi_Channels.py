# %%
import torch
from torch import nn
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l


# %%
# 多通道输入，结果加到一个通道上
# 此处有图，详情见书
# https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter05_CNN/5.3_channels
# https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.3_conv_multi_in.svg
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    # 其实就在每个维度上搞一个for循环，然后最后加在一起
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res


X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

corr2d_multi_in(X, K)

# %%

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])


# 多通道输出
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


K = torch.stack([K, K + 1, K + 2])
print(K.shape)  # torch.Size([3, 2, 2, 2])

corr2d_multi_in_out(X, K)


# %%
# 1*1卷积
# 图片说明：
# https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter05/5.3_conv_1x1.svg
# 这里上下两个卷积，上面的浅色的跟三层输入相乘加起来（加起来指5.3.1那么算）得到通道中前面的浅色输出，下面的深色的跟三层输入相乘加起来得到输出通道中后面的深色输出
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.view(c_i, h * w)
    K = K.view(c_o, c_i)
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

print((Y1 - Y2).norm().item() < 1e-6)
