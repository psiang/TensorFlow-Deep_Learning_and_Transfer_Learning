# 模型搭建教程

本教程将简要介绍几种经典的CNN模型，并提供在TensorFlow2.0中的实现方式，大家可以以此为例实现其他CNN模型的构建。

TensorFlow2.0中已经内置了keras，keras可以方便地将模型构建出来。Model和Layer类型的具体操作参阅[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers)，此处不赘述。所有的模型搭建函数将返回一个搭建好的Model类型。

参考文献：[keras实现常用深度学习模型](https://blog.csdn.net/wmy199216/article/details/71171401)。

## 目录

- [LeNet](#LeNet)
- [AlexNet](#AlexNet)
- [VGGNet](#VGGNet)
- [GoogLeNet](#GoogLeNet)
- [ResNet](#ResNet)

## LeNet

![LeNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/LeNet.png)

### LeNet模型介绍

LeNet来自论文[Gradient-Based Learning Applied to Document Recognition](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf)。如上图所示，它的结构比较简单：

**卷积1 - 池化1 -  
卷积2 - 池化2 -  
全连接1（卷积） - 全连接2 - 输出层**

论文中使用的是tanh激活函数，输出层论文采用的是Guassian Connection。另外根据论文**全连接1应该做卷积操作Conv而不是传统意义上的全连接Dense**。

### LeNet模型实现

在这里，池化我们采用MaxPool，同时激活函数使用Relu，输出层采用Softmax，并简化全连接层为一层，由此可以构建以下模型和代码：

**卷积1 - 池化1 -  
卷积2 - 池化2 -  
全连接2 - softmax输出层**

TIPS：如果需要复现LeNet5，只需添加一层全连接1即可。

```python
# LeNet建模
def LeNet(input_shape, output_shape):
    # 设置模型各层（卷积-池化-卷积-池化-全连接）
    model = Sequential([
        # 第一层（卷积加池化）
        Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',
               kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # 第二层（卷积加池化）
        Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # 第三层（全连接）
        Flatten(),
        Dense(100, activation='relu'),
        # 第四层（全连接输出）
        Dense(output_shape, activation='softmax')
    ])
    return model
```

## AlexNet

![AlexNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/AlexNet.png)

### AlexNet模型介绍

AlexNet来自论文[ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)。如上图所示，大致上它分了八层，但是它分成了**两个GPU**来构建，这两个部分是并行运行的，最后在输出层汇总：

**卷积1 - 池化1 -  
卷积2 - 池化2 -  
卷积3 - 卷积4 - 卷积5 -  
 全连接1 - 全连接2 - 输出层**

其中**从“卷积1”到“全连接2”都是分成两个GPU分别运行的**，除了“卷积3”和两次全连接处理数据都只和**同一个GPU**的前一层核映射相连接。

由于该模型在全连接时形成的参数过多，所以论文中使用**Dropout方法隐去一些节点减少计算量**，隐去比例论文给的是0.5。在论文中池化采用MaxPool，同时激活函数使用Relu，输出层采用Softmax。具体核大小和步长可以参见论文3.5节。

### AlexNet模型实现

由于测试时没有使用两个GPU，且为了方便学习，此处简化了模型，**并没有分成两个部分来分别构建**，而是直接两个部分一起，相当于处理数据和两个GPU的前一层核映射都相连接，GPU就不作区分了。

TIPS：如果想用tensorflow2.0原汁原味地实现AlexNet的双GPU，可以参见官网提供的教程[分布式训练](https://www.tensorflow.org/tutorials/distribute/keras)。同时对GPU进行更细粒度的控制也可以参见官网文档[使用GPU](https://www.tensorflow.org/guide/gpu)。

```python
# AlexNet建模
def AlexNet(input_shape, output_shape):
    # 设置模型各层
    model = Sequential([
        # 第一层
        Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape, padding='valid', activation='relu',
               kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # 第二层
        Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu',
               kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # 第三至五层
        Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # 第六至八层
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    return model
```

## VGGNet

![VGGNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/VGGNet.png)

### VGGNet模型介绍

VGGNet来自论文[Very Deep Convolutional Networks for Large Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf%20http://arxiv.org/abs/1409.1556)。论文阐述了6种VGG模型，VGG16是其中一种16层（没把池化算上）的VGG模型，这里以此为例。如上图所示，它的结构就是简单的卷积叠加：

**卷积 1 - 卷积 2 - 池化 1 -  
卷积 3 - 卷积 4 - 池化 2 -  
卷积 5 - 卷积 6 - 卷积 7 - 池化 3 -  
卷积 8 - 卷积 9 - 卷积10 - 池化 4 -  
卷积11 - 卷积12 - 卷积13 - 池化 5 -  
全连接 1 - 全连接 2 - 输出层**

输出层为Softmax，池化采用MaxPool，激活函数采用Relu。

### VGGNet模型实现

按照定义构建即可，注意VGG会耗费更多计算资源，并且使用了更多的参数，导致更多的内存占用，可能会造成**内存溢出**使得程序崩溃，在model.fit的时候应**适当调节batch_size参数**。同时也是因为参数过多，**训练十分缓慢**，建议使用一些预训练模型。

本项目在训练时这个模型进行场景分类时表现并不太好，本项目只训练了20epochs，可能是因为训练轮次太少了，网络还没有收敛。具体原因待探究。

```python
# VGG16
def VGG16(input_shape, output_shape):
    model = Sequential([
        # block1
        Conv2D(64, (3, 3), strides=(1, 1), input_shape=input_shape, padding='same', activation='relu',
               kernel_initializer='uniform'),
        Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # block2s
        Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # block3
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # block4
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # block5
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
        MaxPooling2D(pool_size=(2, 2)),
        # 全连接
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    return model
```

Keras**自带了原版VGG16的代码**，可以引入tensorflow.keras.applications.vgg16直接使用，就是输入输出层的大小要改一下。除此之外还提供了VGG19的模型。下面的代码构建的模型和上面的是**等效**的：

```python
# VGG16
def VGG16(input_shape, output_shape):
    # 为了改输出层大小得到去掉原版全连接的vgg16
    model = vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape)
    # 加上全连接
    tensor = Flatten(name='flatten')(model.output)
    tensor = Dense(4096, activation='relu', name='fc1')(tensor)
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(4096, activation='relu', name='fc2')(tensor)
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(output_shape, activation='softmax')(tensor)
    # 搭建模型
    model_vgg = Model(inputs=model.input, outputs=tensor, name='vgg16')
    return model_vgg
```

## GoogLeNet

![GoogLeNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/GoogLeNet.png)

### GoogLeNet模型介绍

GoogLeNet，最早版本来自[Going deeper with convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)。这个模型在上图乍一看流程挺复杂，但是它其实是由**多个相似结构**组成，相似结构如下图所示：

![Inception](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/Inception.png)

这个相似结构被称作**Inception**。这样一来就可以将GoogLetNet的结构简洁地表示成：

**卷积1 - 池化1 - LRN1 -  
卷积2 - 卷积3 - LRN2 - 池化2 -  
Inception1 - Inception2 - 池化3 -  
Inception3 - Inception4 - Inception5 - Inception6 - Inception7 - 池化4 -  
Inception8 - Inception9 - 池化5 -  
全连接 - 输出层**

其中**除了“池化5”为AveragePool**外，其他池化均为MaxPool。输出层为Softmax，激活函数采用Relu。

上面介绍的为最早的GoogLeNet，又称为Inception v1。后面又对Inception的结构进行了优化，出现了Inception v2、v3、v4等不同版本。

### GoogLeNet模型实现

项目的具体实现给每个卷积都加了一个Batch Normalization层，并为Inception封装了一个接口，下面作具体阐述。

另外，Keras提供了[Inception v3](https://keras.io/applications/#inceptionv3)的模型可直接使用。

#### Batch Normalization实现

从AlexNet开始就介绍了一种对数据归一化的优化方式LRN *(Local Response Normalization)* ，但是效果似乎没有BN *(Batch Normalization)* 好，所以**项目实现时采用BN而不再使用LRN**。

BN提出自论文[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf%20http://arxiv.org/abs/1502.03167)，论文详细阐述了为什么要归一化以及归一化的作用，此处不作讨论。

BN在应用时应该放在**每个卷积Conv之后**，激活函数Relu之前（也有些资料认为应放在激活函数之后），带BN层的卷积实现如下：

```python
# 带BN层的卷积
def __Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    # 卷积
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides)(x)
    # 归一化
    x = BatchNormalization(axis=3)(x)
    # 激活函数
    x = Activation('relu')(x)
    return x
```

#### Inception实现

按照Inception的构造实现即可，与最初的Inception相比对每个卷积都增加了BN层，代码如下：

```python
# Inception v1
def __Inception_v1(x, nb_filter):
    # 分支1
    branch1x1 = __Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    # 分支2
    branch3x3 = __Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch3x3 = __Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
    # 分支3
    branch5x5 = __Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    branch5x5 = __Conv2d_BN(branch5x5, nb_filter, (5, 5), padding='same', strides=(1, 1), name=None)
    # 分支4
    branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branchpool = __Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
    # 合并分支
    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
    return x
```

#### 最终GoogLeNet的实现

按照GoogLeNet的结构实现即可。

```python
# GoogLeNet
def GoogLeNet(input_shape, output_shape):
    # 输入层
    inpt = Input(shape=input_shape)
    # 卷积1 - 池化1
    x = __Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # 卷积2 - 卷积3 - 池化2
    x = __Conv2d_BN(x, 192, (1, 1), strides=(1, 1), padding='same')
    x = __Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Inception1 - Inception2 - 池化3
    x = __Inception_v1(x, 64)  # 256
    x = __Inception_v1(x, 120)  # 480
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Inception3 - Inception4 - Inception5 - Inception6 - Inception7 - 池化4
    x = __Inception_v1(x, 128)  # 512
    x = __Inception_v1(x, 128)
    x = __Inception_v1(x, 128)
    x = __Inception_v1(x, 132)  # 528
    x = __Inception_v1(x, 208)  # 832
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Inception8 - Inception9 - 池化5
    x = __Inception_v1(x, 208)
    x = __Inception_v1(x, 256)  # 1024
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    # 全连接 - 输出层
    x = Dropout(0.4)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(output_shape, activation='softmax')(x)
    model = Model(inpt, x, name='googlenet')
    return model
```

## ResNet

![ResNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/ResNet.png)

### ResNet模型介绍

ResNet来自论文[Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)。论文提出了5种ResNet，在这里以其中34层的模型ResNet34为例进行介绍。ResNet34的结构如上图所示，虽然层数很多，但是都由同一个**Block形式叠加而成**的。Block的结构如下：

![Block](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/Block.png)

每个Block在**两层卷积之后的结果f(x)与Block的输入x相加合并了**。ResNet34所有Block中卷积核的大小都是3×3的，卷积核数量不同的Block结构组成了ResNet34：

**卷积 - MaxPool池化 -  
Block(64)_1 - Block(64)_2 - Block(64)_3 -  
Block(128)_1 - Block(128)_2 - Block(128)_3 - Block(128)_4 -  
Block(256)_1 - Block(256)_2 - Block(256)_3 - Block(256)_4 - Block(256)_5 - Block(256)_6 -  
Block(512)_1 - Block(512)_2 - Block(512)_3 -  
AvgPool池化 - 输出层**

Block中与输入x的合并要求卷积的结果f(x)和输入x的**维数一致**，不然就没法相加。但其中Block(64)_1、Block(128)_1、Block(256)_1、Block(512)_1的f(x)和x的**维数不一致**（即ResNet图中的虚线），论文中提供了三种解决方案：

1. 用0填充上一层的x使之与f(x)维数一致
2. 用卷积投影维数不一致的Block，其他Block保持不变
3. 用卷积投影所有Block

另外更高层数的ResNet例如ResNet50、ResNet101等对Block有略微不同的定义方式，但是大同小异。

### ResNet模型实现

ResNet对每一个卷积也增加了BN层进行优化，和[上面](#batch-normalization实现)的代码是一样的，这里不再展示。

另外，Keras提供了ResNet50、ResNet101、ResNet152等模型可以直接使用，详情[点击此处](https://keras.io/applications/)。

#### Block的实现

项目实现的时候对于维数不一致的Block采用了第二种解决方案，即用卷积投影成相同的维度。

```python
# Block
def __Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), is_projection=False):
    # 两层卷积
    x = __Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = __Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    # 判断是否需要投影相加
    if is_projection:
        shortcut = __Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x
```

#### 最终ResNet的实现

按照ResNet结构直接实现即可。

```python
# ResNet-34
def ResNet34(input_shape, output_shape):
    inpt = Input(shape=input_shape)
    # 卷积池化
    x = ZeroPadding2D((3, 3))(inpt)
    x = __Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Block(64)
    x = __Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # Block(128)
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), is_projection=True)
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # Block(256)
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), is_projection=True)
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # Block(512)
    x = __Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), is_projection=True)
    x = __Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=inpt, outputs=x)
    return model
```
