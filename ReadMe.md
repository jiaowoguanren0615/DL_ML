#                                   ReadMe

## 项目说明：DL_ML项目仓库中包含机器学习和深度学习脚本

## 分类网络使用的图像数据集下载地址：链接：https://pan.baidu.com/s/1bSQ912IhXaMM1TtIezD_fQ?pwd=0615 
提取码：0615 
--来自百度网盘超级会员V2的分享



## 检测，分割网络使用的图像数据集下载地址：链接：https://pan.baidu.com/s/1AeW_rjhPIDcANoifg7-vAg?pwd=0615 
提取码：0615 
--来自百度网盘超级会员V2的分享



## 机器学习书籍下载地址：链接：https://pan.baidu.com/s/17L2TLq9BDvwrVUOGAHINqw?pwd=0615 
提取码：0615 
--来自百度网盘超级会员V2的分享



## 1. classification项目内含有两个脚本，从数据预处理 ---> 筛选 ---> 建模 ---> 训练 ---> 评估（ROC绘制）并支持DP单机多卡运行，修改参数即可。



## 2. Pre_process_data项目内，help_function.py为机器学习二分类整体流程脚本，multi_classes_clf.py为机器学习多分类整体流程脚本。



## 3. Deep_learning项目内包含图像分类、语义分割、DDP单机多卡运行代码。
    (1)具体GPU资源占用情况都有备注，根据自身实际情况调整参数即可
    (2)项目中的自动混合精度是默认开启的，如果自己的GPU不支持自动混合精度，需要把参数设置为False，支持amp的GPU系列有（RTX，Titan、Tesla等）



## 4. Machine_learning项目内包含对(Adaptive Computation and Machine Learning) Kevin P. Murphy - Probabilistic Machine Learning_ An Introduction-The MIT Press (2022)的代码复现。
