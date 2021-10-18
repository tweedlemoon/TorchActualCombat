# %%
import torch
from torch import nn


# %%
# 池化层也称下采样层，会压缩输入的特征图，一方面减少了特征，导致了参数减少，进而简化了卷积网络计算时的复杂度；
# 另一方面保持了特征的某种不变性（旋转、平移、伸缩等）。

# 此函数定义了一个pool_size大小的池化，X是输入的图片，池化方法默认为max
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    # 池化只需要指定尺寸，并不像卷积核一样要指定数值
    # 由于尺寸缩小原理（填充就是为了抵消这个尺寸缩小的）
    # 所以结果尺寸就是X.shape[0] - p_h + 1, X.shape[1] - p_w + 1，Y即为输出
    Y = torch.zeros(X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                # 1.max pooling
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                # 2.average pooling
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
            else:
                raise ValueError
    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))

print(pool2d(X, (2, 2), 'avg'))

# %%
# 带填充的池化
# X是0-15共16个数，尺寸是1*1*4*4
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
print(X)

# 3的意思是3*3的池化尺寸
pool2d = nn.MaxPool2d(3)
# also
# pool2d = nn.MaxPool2d((3, 3))
print(pool2d(X))

# 填充和步幅
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
print(pool2d(X))

# %%
X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
# 将X和X+1放在一起组成两个通道
X = torch.cat((X, X + 1), dim=1)
print(X)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
