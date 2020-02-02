# 模型搭建教程

本教程将简要介绍几种经典的CNN模型，并提供在tensorflow2.0中的实现方式，大家可以以此为例实现其他CNN模型的构建。

tensorflow2.0中已经内置了keras，keras可以方便地将模型构建出来。Model和Layer类型的具体操作参阅[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers)，此处不赘述。所有的模型搭建函数将返回一个搭建好的Model类型。

参考文献：[keras实现常用深度学习模型](https://blog.csdn.net/wmy199216/article/details/71171401)。

## 目录

- [LeNet](#LeNet)
- [AlexNet](#AlexNet)

## LeNet

![LeNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/LeNet.png)

### LeNet模型介绍

LeNet来自论文Gradient-Based Learning Applied to Document Recognition。如上图所示，它的结构比较简单：

卷积1 - 池化1 - 卷积2 - 池化2 - 全连接1（卷积） - 全连接2 - 输出层

论文中使用的是tanh激活函数，输出层论文采用的是Guassian Connection。另外根据论文**全连接1应该做卷积操作Conv而不是传统意义上的全连接Dense**。

### LeNet模型实现

在这里，池化我们采用MaxPooling，同时激活函数使用Relu，输出层采用softmax，并简化全连接层为一层，由此可以构建以下模型和代码：

卷积1 - 池化1 - 卷积2 - 池化2 - 全连接2 - softmax输出层

TIPS：如果需要复现LeNet5，只需添加一层全连接1即可。

```python
# LeNet建模
def LeNet(input_shape, output_shape):
    # 设置模型各层（卷积-池化-卷积-池化-全连接）
    model = models.Sequential()
    # 第一层（卷积1加池化1）
    model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 第二层（卷积2加池化2）
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # 展平数据
    model.add(layers.Flatten())
    # 第三层（全连接2）
    model.add(layers.Dense(100, activation='relu'))
    # 第四层（softmax输出层）
    model.add(layers.Dense(output_shape, activation='softmax'))
    return model
```

## AlexNet

![AlexNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/AlexNet.png)

### AlexNet模型介绍

AlexNet来自论文ImageNet Classification with Deep Convolutional Neural Networks。如上图所示，大致上它分了八层，但是它分成了**两个GPU**来构建，这两个部分是并行运行的，最后在输出层汇总：

卷积1 - 池化1 - 卷积2 - 池化2 - 卷积3 - 卷积4 - 卷积5 - 全连接1 - 全连接2 - 输出层

其中**从“卷积1”到“全连接2”都是分成两个GPU分别运行的**，除了“卷积3”和两次全连接处理数据都只和**同一个GPU**的前一层核映射相连接。

由于该模型在全连接时形成的参数过多，所以论文中使用**Dropout方法隐去一些节点减少计算量**，隐去比例论文给的是0.5。在论文中池化采用MaxPooling，同时激活函数使用Relu，输出层采用softmax。具体核大小和步长可以参见论文3.5节。

### AlexNet模型实现

由于测试时没有使用两个GPU，且为了方便学习，此处简化了模型，**并没有分成两个部分来分别构建**，而是直接两个部分一起，相当于处理数据和两个GPU的前一层核映射都相连接，GPU就不作区分了。

TIPS：如果想用tensorflow2.0原汁原味地实现AlexNet的双GPU，可以参见官网提供的[分布式训练](https://www.tensorflow.org/guide/distributed_training)。同时对GPU进行更细粒度的控制也可以参见官网文档[使用GPU](https://www.tensorflow.org/guide/gpu)。

```python
# AlexNet建模
def AlexNet(input_shape, output_shape):
    # 设置模型各层
    model = models.Sequential()
    # 第一层
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第二层
    model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第三至五层
    model.add(
        layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(
        layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # 第六至八层
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_shape, activation='softmax'))
    return model
```
