# %%
import torch
from torch import nn
import os

# %%
# 演示Tensor的读写
directory = os.path.abspath(os.path.join(os.getcwd(), 'Datasets', '14_TempSaveData'))
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
