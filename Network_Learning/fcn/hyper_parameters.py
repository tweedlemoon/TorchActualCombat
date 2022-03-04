import os

# dataset paths
# VOCdevkit的文件位置
VOC_Dataset_Root = "/Data20T/data20t/data20t/Liuyifei/Datasets"
# VOC2012的文件位置
VOC2012_Dataset_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets/VOCdevkit/VOC2012"
# FCNCOCOResnet训练好的pth位置
FCN_ResNet_COCOTrained_Path = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained"
FCN_ResNet50_COCO = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained/fcn_resnet50_coco.pth"
FCN_ResNet101_COCO = "/Data20T/data20t/data20t/Liuyifei/Datasets/fcn_trained/fcn_resnet101_coco.pth"
# 结果存放位置
if not os.path.exists("./results"):
    os.mkdir("./results")
Result_Path = "./results"

# parameters
# 使用哪个GPU
Which_GPU = "1"
# 最初学习率
Initial_Learning_Rate = 0.0001
# batchsize
Batch_Size = 16
# Epoch
Epoch = 4
# 打印的频率
Print_Frequency = 10
