import matplotlib.pyplot as plt
import os
import sys
import PIL
import cv2
import torch
from torchvision import transforms

from Toolkits.hyper_parameters import *


# 展示一张图片
def show_one_pic(pic):
    plt.imshow(pic)
    plt.show()


# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([transforms.ToTensor(), ])
unloader = transforms.Compose([transforms.ToPILImage, ])
# unloader = transforms.ToPILImage()

# %%
# 一张桌子的图片，334*500的图片
filename = "2007_001834"
a_picture = os.path.join(VOC2012_Pic, filename) + ".jpg"
a_picture_mask = os.path.join(VOC2012_Mask, filename) + ".png"

# %%
# 3种读图片的方式，PIL是一个复杂的数据结构，而cv和plt读图片则都是简单的ndarray
img_PIL = PIL.Image.open(a_picture)
img_PIL_m = PIL.Image.open(a_picture_mask)
print(type(img_PIL))
# opencv存成的是BGR(反着来的)
# 使用cv2.cvtColor(图像名, cv2.COLOR_BGR2RGB)方法可以把它转回去
img_opencv = cv2.imread(a_picture)
img_opencv_m = cv2.imread(a_picture_mask)
print(type(img_opencv))
# plt.imread读取的确实是RGB，但是归一化处理了，即除以了255
img_plt = plt.imread(a_picture)
img_plt_m = plt.imread(a_picture_mask)
print(type(img_plt))

# 具体如下所示
a = img_plt[100][100]
# 此像素点为RGB(192,128,0)土黄色
# imgplt 0.75294,0.50196,0
# imgopencv 0,128,192

plt1 = plt.subplot(321)
plt1.imshow(img_plt)
plt1.set_title('img_plt')
plt2 = plt.subplot(322)
plt2.imshow(img_plt_m)
plt2.set_title('img_plt_m')
# 转回去
img_opencv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
plt3 = plt.subplot(323)
plt3.imshow(img_opencv)
plt3.set_title('img_opencv')
plt4 = plt.subplot(324)
plt4.imshow(img_opencv_m)
plt4.set_title('img_opencv')
plt5 = plt.subplot(325)
plt5.imshow(img_PIL)
plt5.set_title('img_PIL')
plt6 = plt.subplot(326)
plt6.imshow(img_PIL_m)
plt6.set_title('img_PIL_m')
plt.show()
pass

# %%
# 我们这里比较喜欢使用PIL读取图片与tensor进行转接
img_PIL = PIL.Image.open(a_picture).convert('RGB')
# img_tensor = loader(img_PIL)
# img_PIL = unloader(img_tensor)
