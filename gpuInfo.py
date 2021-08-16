import torch

# 判定cuda是否可用
available = torch.cuda.is_available()
# gpu数量
gpu_num = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_num > 0) else "cpu")
device_name = torch.cuda.get_device_name(device)

if __name__ == '__main__':
    print('cuda是否可用：', available)
    # print('gpu数量：', gpu_num)
    print('设备名称：', device_name)
    # 做一个简单运算
    print(torch.rand(3, 3).cuda())
