# %%
import torch
from torch import nn


# %%
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x - x.mean()


# 实例化这个层，然后做前向计算
layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))

# %%
# Linear输入第一个是feature数第二个是label数
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
print(net)
y = net(torch.rand(4, 8))
y.mean().item()


# %%
# 除了像4.2.1节那样直接定义成Parameter类外，还可以使用ParameterList和ParameterDict分别定义参数的列表和字典。
class MyListDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            # 复习：torch.mm是矩阵线代乘（正常乘）
            x = torch.mm(x, self.params[i])
        return x


net = MyListDense()
print(net)


# %%
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
            'linear1': nn.Parameter(torch.randn(4, 4)),
            'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})  # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


net = MyDictDense()
print(net)

# %%
x = torch.ones(1, 4)
print(net(x, 'linear1'))
print(net(x, 'linear2'))
print(net(x, 'linear3'))

# %%
net = nn.Sequential(
    MyDictDense(),
    MyListDense(),
)
print(net)
print(net(x))
