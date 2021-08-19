'''
这里是有关pytorch Tensor的实例以及说明
'''

'''
pytorch将int, float, int array, float array都进行了向量化（Tensor）
而string则没有对应的数据类型，只能使用自然语言处理中的一些方法进行编码，如独热编码、Embedding如word2vec和glove等
'''

# %%
import torch
import numpy as np

# %%


# float = float32
# double = float64
# half = float16

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 类型是torch.float，torch.float只是表明类型，并不能调用
a = torch.float
# [0.,0.,0.]
# cpu
a = torch.FloatTensor(3)
# gpu
a = torch.cuda.FloatTensor(3)

# 有以下几种
# torch.FloatTensor：32bit float
# torch.DoubleTensor：64bit float
# torch.HalfTensor：16bit float
# torch.ByteTensor：8bit usigned integer
# torch.CharTensor：8bit signed integer
# torch.ShortTensor：16bit int
# torch.IntTensor：32bit int
# torch.LongTensor：64bit int

# %%

a = torch.randn(2, 3)
# a.type()来看
print('a =', a, '\n', 'a_TYPE =', a.type())
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.cuda.FloatTensor))

a = torch.randn(2, 3).cuda()
print('a =', a, '\n', 'a_TYPE =', a.type())
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.cuda.FloatTensor))

# %%

# 从CPU转化成GPU
a = torch.randn(2, 3)
print('a =', a, '\n', 'a_TYPE =', a.type())
a = a.cuda()
print('a =', a, '\n', 'a_TYPE =', a.type())

# %%
# python变量转pytorch变量，注意要小写，直接转tensor，会根据变量不同而转变不同
a = 1
a = torch.tensor(a)
print('a =', a, '\n', 'a_TYPE =', a.type())
# 这里shape是0维，这是pytorch0.3以后的特性，标量视为0维，向量视为1维，以此类推
print('a_SHAPE = ', a.shape, ' a_SIZE = ', a.size())

a = 1.0
a = torch.tensor(a)
print('a =', a, '\n', 'a_TYPE =', a.type())
print('a_SHAPE = ', a.shape, ' a_SIZE = ', a.size())

a = [[1.1, 1.2], [1.1, 1.2], [1.1, 1.2]]
# torch.Tensor(里面放数据即列表（不推荐），或者规格shape)
# torch.tensor(里面只能放数据即列表)
a = torch.Tensor(a)
print('a =', a, '\n', 'a_TYPE =', a.type())
print('a_SHAPE = ', a.shape)

a = torch.Tensor(2, 3)
print('a =', a, '\n', 'a_TYPE =', a.type())
print('a_SHAPE = ', a.shape)

# %%
# numpy加入
# 2*3个1
a = np.ones([2, 3])
print(a)
a = torch.from_numpy(a)
print('a =', a, '\n', 'a_TYPE =', a.type(), 'a_SHAPE = ', a.shape)
# 仅仅一个向量
a = np.array([2, 3])
print(a)
a = torch.from_numpy(a)
print('a =', a, '\n', 'a_TYPE =', a.type(), 'a_SHAPE = ', a.shape)

# %%
# 1维度适合表示bias和linear input

# 2维度适合表示linear input batch
a = torch.randn(2, 3)
print(a)
print(a.shape)
print(a.size(0))
print(a.size(1))
print(a.shape[0])
print(a.shape[1])

# 3维度适合RNN
a = torch.rand(4, 2, 3)
print(a)
print(a.size())
print(a.size(0))
print(a.size(1))
print(a.size(2))

# 4维度适合CNN
# 28*28,3表示rgb3通道，如果是黑白就1，2表示两张图片丢进去
a = torch.rand(2, 3, 28, 28)
print(a)
print(a.size())

# %%
a = torch.rand(2, 3, 28, 28)
# number of element表示占用内存空间大小
print(a.numel())
# 维度
print(a.dim())
