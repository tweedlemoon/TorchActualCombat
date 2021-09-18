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

# 这也是初始化经常使用的方法
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

# %%
# 初始化
a = torch.rand(2, 3, 4)
print('a =', a, '\n', 'a_TYPE =', a.type())
b = torch.rand_like(a)
print('b =', b, '\n', 'b_TYPE =', b.type())
# randint(最小值，最大值，tensor尺寸列表)
a = torch.randint(1, 10, [3, 3])
print('a =', a, '\n', 'a_TYPE =', a.type())

# %%
a = torch.randn(1, 2, 3, 4)
print(a)
# 直接索引
print(a[0])
print(a[0, 1])
print(a[0, 1, 2])
print(a[0, 1, 2, 1])
print(a[0, 1, 2, ::2])
print(a.index_select(2, torch.tensor([0, 1])))
print(a.index_select(2, torch.arange(2)))
print(a[0, 1, ...])

print(a[0, :, :, 2])
print(a[0, ..., 2])

# %%
x = torch.randn(3, 4)
print(x)
mask = x.ge(0)
print(mask)
result = torch.masked_select(x, mask)
print(result)

# %%
x = torch.randn(5, 3)
print(x)
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print(x.size(), y.size(), z.size())
print(y)
print(z)

# %%
x = torch.randn(5, 3)
y = x.view(15)
print(x)
print(y)
x += 1
print(x)
print(y)  # 也加了1

# %%
x = torch.randn(5, 3)
x_cp = x.clone().view(15)
print(x)
print(x_cp)
x -= 1
print(x)
print(x_cp)

# %%
# arange(start,end,step)
x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x + y)

# %%
x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y = y + x
print(id(y) == id_before)  # False

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
id_before = id(y)
y[:] = y + x
# also
# y += x
# torch.add(x, y, out=y)
# y.add_(x)
print(id(y) == id_before)  # True

# %%
a = torch.ones(5)
b = a.numpy()
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
a += 1
print(a, b)
b += 1
print(a, b)

# %%
# 以下代码只有在PyTorch GPU版本上才会执行
if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU
    y = torch.ones_like(x, device=device)  # 直接创建一个在GPU上的Tensor
    x = x.to(device)  # 等价于 .to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # to()还可以同时更改数据类型

# %%
x = torch.ones(2, 2, requires_grad=True)
print(x)
print(x.grad_fn)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
# z.mean就是返回矩阵所有元素加和的平均值
out = z.mean()
print(z, out)

print(out)
print(x)
out.backward()  # 等价于 out.backward(torch.tensor(1.))
print(x.grad)

# 再来反向传播一次，注意grad是累加的
out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

# %%
a = torch.randn(2, 2)  # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)

# %%
x = torch.Tensor([1, 2, 3, 4])
# x = torch.tensor([1, 2, 3, 4], requires_grad=True)
x.requires_grad_(True)
y = 2 * x * x
print(y)
z = y.sum()
try:
    x.grad.data.zero_()
except:
    pass
z.backward()
print(x.grad)

# %%
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print(z)
# 现在 z 不是一个标量，所以在调用backward时需要传入一个和z同形的权重向量进行加权求和得到一个标量。
v = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float)
z.backward(v)
print(x.grad)

# %%
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2

# 原本为y3=x^2+x^3
# 但是由于y2被阻断了，所以计算梯度时将y2刨除在外（类似于把y2当成了常量）
# 结论就是2

print(x.requires_grad)
print(y1, y1.requires_grad)  # True
print(y2, y2.requires_grad)  # False
print(y3, y3.requires_grad)  # True

y3.backward()
print(x.grad)

# %%
x = torch.ones(1, requires_grad=True)

print(x.data)  # 还是一个tensor
print(x.data.requires_grad)  # 但是已经是独立于计算图之外，所以此处应为false

y = 2 * x
x.data *= 100  # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x)  # 更改data的值也会影响tensor的值
print(x.grad)
