'''
mnist_project
使用三层神经网络解决MNIST问题
课程跟随练习代码
'''

# torch常用导入
import torch
from torch import nn
from torch.nn import functional as F
# torch相关工具包
from torch import optim
import torchvision

from matplotlib import pyplot as plt

# 包
from mnist_project.utils import *

# batch_size为一次处理图片的数量，充分利用cpu能力，从而降低处理时间
batch_size = 512

'''
1. 加载数据集
70k图片
60ktraining
10ktest
'''

train_load = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data/',
    train=True,
    download=True,
    # 下载下来的是numpy格式，先ToTensor
    # 然后使用正则化，让数据集尽量均匀分布在0附近，不正则效果会差点
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size,
    shuffle=True)

test_load = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data/',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size,
    shuffle=False)

# 显示加载图片
x, y = next(iter(train_load))
print(x.shape, y.shape)
plot_image(x, y, 'sample')

'''
2. 创建神经网络
本例子中使用三层神经网络
'''


class Net(nn.Module):
    # 初始化
    def __init__(self):
        super(Net, self).__init__()
        # w1x+b1
        # linear就是线性层
        # 784->256（隐层从784乘一个矩阵变成256）这个256和下面的64都是由调参经验决定的
        self.fc1 = nn.Linear(784, 256)
        # 256->64
        self.fc2 = nn.Linear(256, 64)
        # 最后一层一定是10，因为是10分类问题
        self.fc3 = nn.Linear(64, 10)

    # 计算过程
    def forward(self, x):
        # x:[batch_size, 1, 28, 28]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# 实例化
net = Net()

# 优化器
# 学习率
learning_rate = 0.01
momentum = 0.9
# net.parameters()返回上述定义神经网络中的所有变量[w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# 保存用于可视化
train_loss = []

# 循环迭代次数
iteration_num = 3
for epoch in range(iteration_num):
    for batch_idx, (x, y) in enumerate(train_load):
        # x: [batch, 1, 28, 28]，y: [batch]
        # 将x打平x: [batch, 1, 28, 28]->[batch, 784]
        x = x.view(x.size(0), 784)

        # 输入网络，由[batch, 784]->[batch, 10]
        out = net(x)
        y_one_hot = one_hot(y)

        # loss，这里是min square error即均方误差
        loss = F.mse_loss(out, y_one_hot)

        # 清零梯度
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 计算w' = w - lr * grad，并更新梯度
        optimizer.step()

        # train_loss记录结果
        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print('epoch:', epoch, 'batch_idx:', batch_idx, 'loss = ', loss.item())

plot_curve(train_loss)

'''
3.准确度测试
使用test_load中的数据进行测试
'''
total_correct = 0
total_samples = len(test_load.dataset)

for x, y in test_load:
    x = x.view(x.size(0), 784)
    out = net(x)
    # 把10维矩阵中最大的那个值设为预测结果
    predict = out.argmax(dim=1)
    # eq是equal，这句话是把每个test_load里的数据都丢网络里，然后算出正确的数量
    correct = predict.eq(y).sum().float().item()
    total_correct += correct

acc = total_correct / total_samples
print('accuracy:', acc)

# 取6张图看看怎么样
x, y = next(iter(test_load))
out = net(x.view(x.size(0), 784))
predict = out.argmax(dim=1)
plot_image(x, predict, 'Recognition Result')
