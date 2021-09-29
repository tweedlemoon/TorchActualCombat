# %%
import torch
from torch import nn
from torch.nn import init

# %%
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化
print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

# 访问模型的参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())

# 一个意思，访问Sequense中第0位网络中的变量
for name, param in net[0].named_parameters():
    print(name, param.size(), type(param))


# %%
# 这个例子的意思是，一定是nn.Parameter才会被添加到变量区，一个普通的tensor是没有这个资格的
# nn.Parameter是Tensor的子类，其本质就是tensor
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        self.weight2 = torch.rand(20, 20)

    def forward(self, x):
        pass


n = MyModel()
for name, param in n.named_parameters():
    print(name)

# 对Tensor的操作都可以对它做
weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad)  # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)

# %%
net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))
# 使用均值0标准差0.01的初始化方法
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

# 选择全零的初始化方法
for name, param in net.named_parameters():
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)


# 自己定义初始化方法然后进行初始化
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

# 通过改变这些参数的data来改写模型参数值同时不会影响梯度，见之前
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)

# %%
# 在有些情况下，我们希望在多个层之间共享模型参数。
# 4.1.3节提到了如何共享模型参数: Module类的forward函数里多次调用同一个层。
# 此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的，下面来看一个例子:

# 这是一个不带偏倚的线性模型
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)
print(net)
for name, param in net.named_parameters():
    # 初始化成固定值3
    init.constant_(param, val=3)
    print(name, param.data)

print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

# 因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)  # 单次梯度是3，两次所以就是6

# %%
# 所以要避免上述情况，则把两个线性层命名为不一样的，否则他们会在内存里指向同一个位置
# 这是一个不带偏倚的线性模型
linear1 = nn.Linear(1, 1, bias=False)
linear2 = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear1, linear2)
print(net)
for name, param in net.named_parameters():
    # 初始化成固定值3
    init.constant_(param, val=3)
    print(name, param.data)

# 由于两个模型并不相同，所以二者的id不同
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))

# 因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的
x = torch.ones(1, 1)
y = net(x).sum()
print(y)
y.backward()
print(net[0].weight.grad)  # 这次梯度正常为3
