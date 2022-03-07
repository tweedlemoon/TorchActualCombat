import os

import torch.utils.data as data
import torchvision
from PIL import Image


# 把下面这行打开，即可看到pytorch封装的VOCSegmentation类，这里把它重写了
# voc_train_dataset = torchvision.datasets.VOCSegmentation(root='', year='2012', image_set='train', )
# 重写的要求：1. 将数据存成一个list能随时找到，label也要对应存好
# 2. 必须有__getitem__函数和__len__函数，分别返回一条数据和数据集整体长度


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        """
        这个是VOCSegmentation的重写，具体见原来的VOCSegmentation类
        :param voc_root:VOCdevkit的上级目录，此路径下一定要有VOCdevkit文件夹
        :param year:2007或者2012
        :param transforms:图像尺寸更改函数，用于对图像进行预先处理，比如尺寸不够、尺寸过大等等
        :param txt_name:训练集的索引txt名称，一般位于/VOCdevkit/VOC2012/ImageSets/Segmentation那里
        """
        super(VOCSegmentation, self).__init__()
        # 断言年份必须是2007或者2012，这是pascal数据集的特性
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 拼合路径VOC2012/2007
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # 断言检查路径是否合法
        assert os.path.exists(root), "path '{}' does not exist.".format(root)

        # 图片路径、遮罩路径、索引txt路径
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        # 断言检查路径是否合法
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)

        # 一行一行读取txt中的数据，并删去每行数据结尾的换行符
        # strip()函数是copy一份字符串，然后返回那个字符串去掉首尾空格和换行（非字符串中间的空格）或指定字符
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 给每个图片和遮罩名字
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        # 断言二者必须相等
        assert (len(self.images) == len(self.masks))

        self.transforms = transforms

    def __getitem__(self, index):
        # 如果要重写VOCSegmentation类，则__getitem__和__len__必须重写，否则报错
        """
        getitem函数的意义就是给一个index能取出数据集里的那个item
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # 图片要转成RGB，而目标则只是mask
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
