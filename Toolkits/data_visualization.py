import sys

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from Toolkits.hyper_parameters import *

sys.path.append("..")


# 展示一张图片
def show_one_pic(pic):
    plt.imshow(pic)
    plt.show()


# %%
# 一张桌子的图片，334*500的图片
filename = "2007_001834"
a_picture = os.path.join(VOC2012_Pic, filename) + ".jpg"
a_picture_mask = os.path.join(VOC2012_Mask, filename) + ".png"

# %%

# # 使用CV2测试一下输出
# # flags 1是rgb图 0是灰度图 -1是原图
# input_pic_cv2 = cv2.imread(input_pic_path, flags=1)
# cv2.imshow('flag=1 rgb pic', input_pic_cv2)
# input_pic_cv2 = cv2.imread(input_pic_path, flags=0)
# cv2.imshow('flag=0 gray pic', input_pic_cv2)
# input_pic_cv2 = cv2.imread(input_pic_path, flags=-1)
# cv2.imshow('flag=-1 origin pic', input_pic_cv2)
# # 等待窗口关闭或任意键输入继续
# cv2.waitKey(0)

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

"""
这里演示matplotlib的用法，在一个窗口中显示多张图片的做法。
"""
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

# %%
"""
关于读入图片，各种形式相互转化的方法
"""
# 我们这里比较喜欢使用PIL读取图片与tensor进行转接
# img_pil = Image.open(a_picture)
# RGB的作用是丢掉第四个通道RGBA中的A，A是透明度
img_pil = Image.open(a_picture).convert('RGB')
# 保存图片
# img_pil.save('路径')

# PIL变numpy，变成的是H*W*C的numpy
img_numpy = np.array(img_pil)
# 注意，ToTensor是H*W*C的numpy转成tensor，如果是PIL进去或者满足np.uint8进去则是正常的C*H*W进去
# 注意，ToPILImage是C*H*W的Tensor或者H*W*C的numpy转成PIL
# 故可以这样写
img_pil = transforms.ToPILImage()(img_numpy)
img_tensor = transforms.ToTensor()(img_pil)
# 或者
img_pil = transforms.ToPILImage()(img_tensor)
img_tensor = transforms.ToTensor()(img_numpy)

# 总结，只有numpy维度是倒着存的
# 只有CV2的RGB是存成GBR的

# cv2的图片展示是最方便的，但是要展示numpy，而且记得要转RGBGBR
cv2.imshow('numpy', cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
