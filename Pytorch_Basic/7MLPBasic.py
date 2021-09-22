# %%
import torch
import numpy as np
import matplotlib.pylab as plt
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l


# %%
# 画出函数的图像并显示
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')


# %%
# ReLU
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
plt.show()

# 看ReLU的梯度（导数）
y.sum().backward()
# y.
xyplot(x, x.grad, 'grad of relu')
plt.show()

# %%
# Sigmoid
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
plt.show()

# 不要忘了梯度清零
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')
plt.show()
