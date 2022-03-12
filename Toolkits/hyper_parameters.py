import os
import platform

# dataset paths
# 系统判定
if platform.system() == "Windows":
    # VOCdevkit的文件位置
    VOC_Dataset_Root = "E:/Datasets/Pascal_voc_2012"
    # VOC2012的文件位置
    VOC2012_Dataset_Path = "E:/Datasets/Pascal_voc_2012/VOCdevkit/VOC2012"

    # 图片的位置
    VOC2012_Pic = os.path.join(VOC2012_Dataset_Path, "JPEGImages")
    # 遮罩的位置
    VOC2012_Mask = os.path.join(VOC2012_Dataset_Path, "SegmentationClass")

    # FCNCOCOResnet训练好的pth位置
    FCN_ResNet_COCOTrained_Path = "E:/Datasets/fcn_trained"
    FCN_ResNet50_COCO = "E:/Datasets/fcn_trained/fcn_resnet50_coco.pth"
    FCN_ResNet101_COCO = "E:/Datasets/fcn_trained/fcn_resnet101_coco.pth"
    # 使用哪个GPU
    Which_GPU = "1"

else:
    # VOCdevkit的文件位置
    VOC_Dataset_Root = "/Data20T/data20t/data20t/Liuyifei/Datasets"
    # VOC2012的文件位置
    VOC2012_Dataset_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets/VOCdevkit/VOC2012"

    # 图片的位置
    VOC2012_Pic = os.path.join(VOC2012_Dataset_Path, "JPEGImages")
    # 遮罩的位置
    VOC2012_Mask = os.path.join(VOC2012_Dataset_Path, "SegmentationClass")

    # FCNCOCOResnet训练好的pth位置
    FCN_ResNet_COCOTrained_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained"
    FCN_ResNet50_COCO = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained/fcn_resnet50_coco.pth"
    FCN_ResNet101_COCO = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained/fcn_resnet101_coco.pth"
    # 使用哪个GPU
    Which_GPU = "1"
