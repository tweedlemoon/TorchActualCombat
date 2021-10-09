# %%
import torch
from torch import nn

# %%
print(torch.cuda.is_available())  # 输出 True
print(torch.cuda.device_count())  # 输出 1
# 查看当前GPU索引号
print(torch.cuda.current_device())  # 输出 0
print(torch.cuda.get_device_name(0))

# %%
# 注意cuda(0)与cuda()等价
x = torch.tensor([1, 2, 3])
print(x)
x = x.cuda(0)
print(x)
print(x.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1, 2, 3], device=device)
# or
# x = torch.tensor([1, 2, 3]).to(device)
print(x)

# %%
# 如果对在GPU上的数据进行运算，那么结果还是存放在GPU上。
y = x ** 2
print(y)

# z = y + x.cpu()  # 会报错

# %%
# 对于模型
net = nn.Linear(3, 1)
print(list(net.parameters())[0].device)

# 转化到GPU上
net.cuda()
print(list(net.parameters())[0].device)

# 输入的值也要在GPU上，不然会报错
x = torch.rand(2, 3).cuda()
net(x)
