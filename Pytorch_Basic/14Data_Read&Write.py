# %%
import torch
from torch import nn
import os

# %%
# 演示Tensor的读写
directory = os.path.abspath(os.path.join(os.getcwd(), '..', 'Datasets', '14_TempSaveData'))
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
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer.state_dict()

# %%
# torch.save(model.state_dict(), PATH)  # 推荐的文件后缀名是pt或pth
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
