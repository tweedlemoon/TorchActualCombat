# %%
import torch
from torch import nn
import os
import sys

# %%
# 演示Tensor的读写

if sys.platform.startswith('win'):
    # 如果是windows
    directory = os.path.join(os.getcwd(), 'Datasets', '14_TempSaveData')
else:
    # linux
    directory = os.path.join(os.getcwd(), '..', 'Datasets', '14_TempSaveData')

if not os.path.exists(directory):
    os.makedirs(directory)

x = torch.ones(3)
torch.save(x, os.path.join(directory, 'x.pt'))

x2 = torch.load(os.path.join(directory, 'x.pt'))
# x2

y = torch.zeros(4)
torch.save([x, y], os.path.join(directory, 'xy.pt'))
xy_list = torch.load(os.path.join(directory, 'xy.pt'))
# xy_list

torch.save({'x': x, 'y': y}, os.path.join(directory, 'xy_dict.pt'))
xy = torch.load(os.path.join(directory, 'xy_dict.pt'))


# xy

# %%
# 注意，只有具有可学习参数的层(卷积层、线性层等)才有state_dict中的条目。
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


net = MLP()
net.state_dict()

# %%
# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()

# %%
# 保存方式1（仅参数）
# torch.save(model.state_dict(), PATH)  # 推荐的文件后缀名是pt或pth
# 加载方式1
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))

# 保存方式2（整个模型）
# torch.save(model, PATH)
# 加载方式2
# model = torch.load(PATH)

# 第一种实践
X = torch.randn(2, 3)
Y = net(X)

PATH = os.path.join(directory, 'net.pt')
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)
