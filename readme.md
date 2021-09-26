# 深度学习Pytorch实战

## 1. 项目信息

本项目为书本《动手学深度学习》（李沐等著）的实战， [原书官网](http://zh.d2l.ai/) [原书配套Github项目地址](https://github.com/d2l-ai/d2l-zh)。  

原书使用深度学习框架为MXNet，作者[Tangshusen](https://tangshusen.me/)将其使用pytorch重构，并将本项目上传至[Github](https://github.com/ShusenTang/Dive-into-DL-PyTorch)。  

我将[本书](https://tangshusen.me/Dive-into-DL-PyTorch/#/)的电子版以及上述Github代码进行学习与糅合，加入了我自己的注释（因为本人水平较低），得到此项目，算作是我的学习笔记。

## 2. 项目结构

1. d2lzh_pytorch

   原书相关的工具箱，其中每一个函数都在学习的过程中有所解释，当在学习的代码中看到这样一条注释

   ```
   # 本函数已保存在d2lzh_pytorch包中方便以后使用。
   ```

   即表示它已经封装在工具包中了。

2. Datasets

   数据集，代码中本身并不包含，在进行初始化后出现，一般是公共的数据集，服务器一般在国外，不想龟速下载的话**记得翻墙**。

3. mnist_project

   ~~已废弃~~，这是在最初的时候接触所写的一个mnist项目代码，在学习中已经有更好的方法实现。

4. notebook和Pytorch_Basic

   是主体代码，前者是jupyter notebook文件，后者是纯代码的py文件，二者几乎没有区别，可以看到一个py对应一个ipynb。

## 3. 如何使用

温馨提示：请打开[原书](https://tangshusen.me/Dive-into-DL-PyTorch/#/)和notebook同步学习，不然你会看不懂的~  

### 1. 环境需求

先装好环境，本项目使用的python版本是3.7，目前最新的是3.7.11，需要以下环境

1. pytorch以及torchtext
2. jupyter
3. matplotlib

#### 附上指令

#### 1. pytorch和torchtext

当没有英伟达GPU时用以下指令（纯CPU）

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

当有英伟达GPU（至少9代以上）可以用GPU计算，需要先装CUDA然后装Pytorch，具体方法建议自行搜索。  

torchtext如下

```
conda install -c pytorch torchtext
```

#### 2. jupyter

```
conda install -c conda-forge jupyterlab
```

#### 3. matplotlib

```
conda install matplotlib 
```

### 2. 项目初始化

下载此项目，**安装Git**，并在你想放置此项目的文件夹下，右键使用Git Bash Here对代码进行克隆

```
$ git clone https://github.com/tweedlemoon/TorchActualCombat.git
```

进入到项目文件夹中，如果你使用pycharm，建议直接右键Open folder as pycharm project  

运行gpuInfo.py查看你的GPU信息

```
$ python gpuInfo.py
```

然后下载数据集

```
$ python dataset_download.py
```

最后开启项目

```
$ jupyter notebook
```

进入notebook文件夹开始跟原书对应进行学习

