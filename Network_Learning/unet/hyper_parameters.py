import os
import platform

# dataset paths
# 系统判定
if platform.system() == "Windows":
    Data_Path = "E:/Datasets/"
    # 使用哪个GPU
    Which_GPU = "0"

else:
    Data_Path = "/root/autodl-tmp"
    # 使用哪个GPU
    Which_GPU = "0"

# 结果存放位置
if not os.path.exists("./results"):
    os.mkdir("./results")
Result_Root = "./results"

# parameters
# 最初学习率
Initial_Learning_Rate = 0.01
# batchsize
# Batch_Size = 16
Batch_Size = 2
# Epoch
Epoch = 200
# 打印的频率
Print_Frequency = 1
