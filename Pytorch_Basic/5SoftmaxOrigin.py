import torch
import torchvision
import numpy as np
import sys
import os

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# %%
# 读数据
directory = os.path.abspath(os.path.join(os.getcwd(), "../", "Datasets"))
if not os.path.exists(directory):
    os.makedirs(directory)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=None, root=directory)

# %%
# 变量初始化
# 输入是784即28的平方
num_inputs = 28 ** 2
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
# 需要记录梯度
W.requires_grad_(True)
b.requires_grad_(True)

# %%
# 插入一条说明，这里是如何进行某个维度的加和并且显示结果
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))


# %%
# 写作softmax函数
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 试一下softmax效果
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(dim=1))


# %%
# 建模
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# %%
# 插入一条说明，这里是如何使用gather函数
# 假设现在有两个样本，做三分类，y_hat是你得到的预测概率
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
# 复习，view是将张量reshape，-1则是可根据已有来推测
# 所以[0,2](1行,2列)进行(-1,1)，即推出(2行,1列)，则会得到y.view(-1, 1)是[[0],[2]]
# gather函数，第一个输入变量是dim，第二个输入变量是一个tensor，输出和这个输入的tensor规格一致
# 用输入tensor的值来做索引替换掉第一个输入变量所在的维得到结果
# 下面这个例子，索引为列（即1），那么输出[0][0]位为y_hat中的[0][0]=0.1，输出[1][0]位为y_hat中的[1][2]=0.5
y_hat.gather(1, y.view(-1, 1))


# 直观看
# [[0.1, 0.3, 0.6],     [0,
#  [0.3, 0.2, 0.5]]      2]  且dim=1
# 输出
# [0.1,
#  0.5]
# 就是把对应位置的值拿出来

# %%
# 损失函数
def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    # 返回负的log(值)，因为是torch.log，所以每行求log，y是真实结果单列化
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


# 准确率
def accuracy(y_hat, y):
    """准确率核心代码"""
    # argmax返回dim=1（列固定）每行的最大值，最大值出现的列如果和真实的y相同则计入
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 上述一个对一个错，准确率应该是0.5
print(accuracy(y_hat, y))


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    """计算准确率"""
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


print(evaluate_accuracy(test_iter, net))

# %%
# 开始训练
num_epochs, lr = 5, 0.1


# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# %%
# 可视化预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
