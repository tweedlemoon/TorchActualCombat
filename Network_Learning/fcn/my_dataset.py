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
        :param voc_root:str VOCdevkit的上级目录，此路径下一定要有VOCdevkit文件夹
        :param year:str 2007或者2012
        :param transforms:function 图像尺寸更改函数，用于对图像进行预先处理，解决比如尺寸不够、尺寸过大等等
        :param txt_name:str 训练集的索引txt名称，一般位于/VOCdevkit/VOC2012/ImageSets/Segmentation那里
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
        :param index: index (int): Index
        :return: tuple: (image, target) where target is the image segmentation.
        """
        # 图片要转成RGB，而目标则只是mask
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """
        len函数就是求该类的length
        :return:int 数据集的长度
        """
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
        :param batch:
        :return:
        """
        # zip把batch元组打包
        images, targets = list(zip(*batch))
        # imgs尺寸统一化处理填0
        batched_imgs = cat_list(images, fill_value=0)
        # mask则填255
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    """
    一个batch中的尺寸统一化
    :param images: Tensor
    :param fill_value: 填充值
    :return: 一个尺寸统一化的batch
    """
    # 计算该batch数据中，channel, h, w的最大值
    # img.shape for img in images这里for img in images是把images[0]作为循环体了
    # 比如images.shape=16*3*375*500（这里图像尺寸一致了，其实不一定一致，这就是它的作用）
    # 即batchsize16，3通道RGB，高度375，宽度500，那么for img in images就是16个循环，把每个图遍历一遍
    # zip将其打包，结果就是(3,3,...)(375,375,...)(500,500,...)各16个，因为zip输入的参数是16个list，每个list里有cwh
    # 接下来一个for循环循环数是3，因为是3个zip，输出是一个迭代器，对应的是3，375和500，然后强制转化成tuple，得到(3,375,500)
    # 所以这两句话的意思就是把这个batch中的最大channel和最大h最大w求出来
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 再把batchsize加上，+是拼接，于是(3,375,500)拼接成了(16,3,375,500)而且这时候后面三项统一为该batch中的最大值了
    batch_shape = (len(images),) + max_size

    # images[0]没有实际意义，本句的意思就相当于new一个新的tensor，大小是batchshape，填充fill_value
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # 然后就是把images里的数据拷进去
    # :img.shape[-2]的意思是0~倒数2位-1，比如:7就是0-6共7个，copy方法就是把每张img拷贝进去
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
