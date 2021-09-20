# %%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch

import d2lzh_pytorch as d2l

# %%

directory = os.path.abspath(os.path.join(os.getcwd(), "../", "Datasets"))
if not os.path.exists(directory):
    os.makedirs(directory)

# directory = os.path.abspath(os.path.join(os.getcwd(), "../Datasets/", "FashionMNIST"))
# if not os.path.exists(directory):
#     os.makedirs(directory)

mnist_train = torchvision.datasets.FashionMNIST(root=directory, train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=directory, train=False, download=True,
                                               transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

print('==================================')

# 可以通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width


# %%
# 本函数已保存在d2lzh包中方便以后使用
# 以下函数可以将数值标签转成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 下面定义一个可以在一行里画出多张图像和对应标签的函数。
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 看一下训练数据集中前10个样本的图像内容和文本标签
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# %%
# 在实践中，数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时
# PyTorch的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取
# 通过参数num_workers来设置4个进程读取数据
# 注意windows由于没有fork()函数导致可能会有问题，所以要判断操作系统是不是windows，然后再读取数据
batch_size = 256
if sys.platform.startswith('win'):
    # 如果是windows，则无法多线程读取
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取一遍训练数据需要的时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
