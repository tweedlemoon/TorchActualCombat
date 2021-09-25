# %%
import torch
from torch import nn
from collections import OrderedDict


# %%
# 使用nn.Module并重构init和forward函数，使得其变成所需的网络
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        # 在python3中可简写为
        # super().__init__()
        # 隐藏层是一个线性的784转256
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        # 一个激活函数
        self.act = nn.ReLU()
        # 输出层是一个线性256转10
        self.output = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


# %%
# 输入一些数据看一下这个网络
X = torch.rand(2, 784)
net = MLP()
print(net)
net(X)


# %%
# module的子类
# Sequential类
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        # 干了这样一件事：
        # 如果传入的是OrderedDict，那就把这个OrderDict拆开拿出里面的modules一个一个加入到self._modules
        # 如果传入的是module，则直接把module加入self._modules
        if len(args) == 1 and isinstance(args[0], OrderedDict):  # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        # 也就是说按照序列一个接一个往前运行
        for module in self._modules.values():
            input = module(input)
        return input


# %%
net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
net(X)

# %%
# 直接使用Sequential
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
net(X)

# %%
# modulelist
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))  # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)


# net(torch.zeros(1, 784)) # 会报NotImplementedError


# %%
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


# %%
# ModuleList不同于普通python的list
class Module_ModuleList(nn.Module):
    def __init__(self):
        super(Module_ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10)])


class Module_List(nn.Module):
    def __init__(self):
        super(Module_List, self).__init__()
        # 区别在这，这里用的是普通的list
        self.linears = [nn.Linear(10, 10)]


net1 = Module_ModuleList()
net2 = Module_List()

print("net1:")
for p in net1.parameters():
    print(p.size())

print("net2:")
for p in net2.parameters():
    print(p)

# %%
# ModuleDict类

net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)  # 添加
print(net['linear'])  # 访问
print(net.output)
print(net)


# net(torch.zeros(1, 784)) # 会报NotImplementedError


# %%
# 一个稍微复杂的模型
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


X = torch.rand(2, 20)
net = FancyMLP()
print(net)
net(X)


class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())

    def forward(self, x):
        return self.net(x)


net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())

X = torch.rand(2, 40)
print(net)
net(X)
