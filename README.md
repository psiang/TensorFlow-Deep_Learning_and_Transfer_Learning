# Scene_Classification

本项目README文档

本项目是为了便于学习场景分类而创建的。本项目中，大家可以学习使用tensorflow2.0搭建CNN模型的方式，同时自己动手进行训练。如果觉得训练需要花费较多时间，也可以使用本项目预训练的模型直接进行测验。

项目主要内容有:

1. 项目的使用教程。
2. 不同CNN模型的搭建。

README文档主要用于介绍项目基本情况、安装及使用环境配置并为其他教程提供索引。

## 目录

- [项目文件结构](#项目文件结构)
- [安装及环境配置](#安装及环境配置)
- [数据来源](#数据来源)
- [项目使用教程](#项目使用教程)
- [模型搭建教程](#模型搭建教程)

## 项目文件结构

```python
.
├── data            '''存放图像数据'''
├── docs            '''存放文档'''
├── model_data      '''存放模型数据'''
|   ├── history     '''存放训练历史数据'''
|   ├── model       '''存放训练模型数据'''
|   └── weight      '''存放训练权值数据'''
├── results         '''存放训练结果'''
|   ├── accuracy    '''准确度图'''
|   ├── loss        '''损失度图'''
|   └── prediction  '''单图像预测图'''
├── rsidea          '''项目代码'''
|   ├── preprocess  '''图像预处理工具'''
|   └── util        '''其他数据处理工具'''
└── __main__.py     '''程序入口'''
```

## 安装及环境配置

测试项目使用的IDE是Pycharm，解释器使用Python3.7。所以下面以此为例介绍安装步骤：

1. Clone项目到本地 <https://github.com/psiang/Scene_Classification>
2. 安装Pycharm <https://www.jetbrains.com/pycharm/>
3. 安装Python3.7 <https://www.python.org/downloads/>
4. 使用Pycharm打开项目
5. 打开 File->Setting->Project Interpreter
6. 解释器设置为Python3.7
7. 至少安装以下包：tensorflow、matplotlib、sklearn、pillow
8. 运行__main__.py中的例子

注意事项：

1. 请务必**安装tensorflow2.0以上**的版本，测试时使用的2.0.1
2. 请务必**不要使用Python3.8**，tensorflow只支持3.5~3.7
3. 可以使用其他支持tensorflow的解释器

## 数据来源

示例数据为**SIRI-WHU Data Set**中12class_tif

图像像素大小为200*200，总包含12类场景图像，每一类有200张，共2400张。

下载地址：<http://www.lmars.whu.edu.cn/prof_web/zhongyanfei/e-code.html>

注意事项：数据中/pond/0002.tif 的大小是190*200，需要注意处理。示例采取了剔除的处理方式。

## 项目使用教程

项目使用教程简要介绍了tensorflow的使用以及其他util的使用。教程将按照以下五个板块介绍：

1. 数据的读取
2. 数据的预处理
3. 模型获取
4. 模型测试
5. 结果图像生成

具体请参见文档：

## 模型搭建教程

模型搭建教程简要介绍了各类CNN模型及其利用tensorflow搭建方式。

具体请参见文档：

参照教程：<https://www.tensorflow.org/tutorials/images/cnn>
