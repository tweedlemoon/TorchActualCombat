# %%
import torch
from torch import nn


# %%
def corr2d(X, K):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
print(corr2d(X, K))

# %%
# 一个边缘检测的简单例子
# 这里x代表一张图片，先1初始化，然后2-6行置零，01交界处视为边缘
X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

# K是一个能识别边缘的卷积核，用X与K进行卷积运算可以看到边缘处不为0而其他地方为0
K = torch.tensor([[1, -1]])
# Y是已经被边缘检测了的输出矩阵
Y = corr2d(X, K)
print(Y)


# 因此得到任务：已知X是一张图片，已知Y是X经过边缘检测的结果，请学习卷积核K

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        # 所以先对K进行初始化
        self.weight = nn.Parameter(torch.randn(kernel_size))
        # 假设有bias，后面可以得到这个bias基本是0
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # 前向传播就是标准的卷积运算
        return corr2d(x, self.weight) + self.bias


# 构造一个核数组形状是(1, 2)的二维卷积层
conv2d = Conv2D(kernel_size=(1, 2))

# 定义下降次数和学习率
step = 20
lr = 0.01
for i in range(step):
    # Y_hat是预测结果
    Y_hat = conv2d(X)
    l = ((Y_hat - Y) ** 2).sum()

    # 反向传播后进行梯度下降
    l.backward()
    # 梯度下降
    conv2d.weight.data -= lr * conv2d.weight.grad
    conv2d.bias.data -= lr * conv2d.bias.grad

    # 别忘了梯度清0
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)

    # 每五次输出一次
    if (i + 1) % 5 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

# 打印出学习到的结果
print("weight: ", conv2d.weight.data)
print("True Kernel:", K)
print("bias: ", conv2d.bias.data)

# 注意，卷积运算和互相关运算实质上不过是上下对称左右对称的运算
# 由于卷积Kernel是学来的，所以无论采取那种运算，学习结果不同，但是最后得到的预测结果都一定是一样的
