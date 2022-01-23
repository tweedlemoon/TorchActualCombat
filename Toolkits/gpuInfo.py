import datetime
import torch
import sys
import os
from Toolkits.emailSender import EmailSender

# 判定cuda是否可用
available = torch.cuda.is_available()
# gpu数量
gpu_num = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu_num > 0) else "cpu")
device_name = torch.cuda.get_device_name(device) if (torch.cuda.is_available() and gpu_num > 0) else "cpu"

if __name__ == '__main__':
    # 定义日志位置
    # 文件名是：时间+当前的文件名去掉后缀+_log.txt
    logName = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S') + '_log.txt'
    logFilePath = os.getcwd() + '/../LogFiles/' + logName
    logFile = open(logFilePath, 'w', buffering=1)
    try:
        # 输出重定向
        # 先把原来的控制台保存一下
        __console__ = sys.stdout
        __error__ = sys.stderr
        # 标准输出和标准错误输出在日志文件中
        sys.stdout = logFile
        sys.stderr = logFile

        print('cuda是否可用：', available)
        # print('gpu数量：', gpu_num)
        print('设备名称：', device_name)
        # 做一个简单运算
        if torch.cuda.is_available():
            result = torch.rand(3, 3).cuda()
            print(result)
        else:
            result = torch.rand(3, 3)
            print(result)
        # 再把控制台和标准错误还原回去
        sys.stdout = __console__
        sys.stderr = __error__
        print('控制台已经还原，程序结束！')
        logFile.close()

        # 发成功邮件
        sender = EmailSender(proName=os.path.basename(sys.argv[0]), logAdd=logFile.buffer.name)
        sender.sendResultEmail()
    except Exception:
        # 发错误邮件
        sender = EmailSender(proName=os.path.basename(sys.argv[0]), logAdd=logFile.buffer.name)
        sender.sendErrorEmail()
