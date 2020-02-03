# 模型搭建教程

本教程将简要介绍几种经典的CNN模型，并提供在TensorFlow2.0中的实现方式，大家可以以此为例实现其他CNN模型的构建。

TensorFlow2.0中已经内置了keras，keras可以方便地将模型构建出来。Model和Layer类型的具体操作参阅[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers)，此处不赘述。所有的模型搭建函数将返回一个搭建好的Model类型。

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

AlexNet来自论文ImageNet Classification with Deep Convolutional Neural Networks。如上图所示，大致上它分了八层，但是它分成了**两个GPU**来构建，这两个部分是并行运行的，最后在输出层汇总：

卷积1 - 池化1 - 卷积2 - 池化2 - 卷积3 - 卷积4 - 卷积5 - 全连接1 - 全连接2 - 输出层

其中**从“卷积1”到“全连接2”都是分成两个GPU分别运行的**，除了“卷积3”和两次全连接处理数据都只和**同一个GPU**的前一层核映射相连接。

由于该模型在全连接时形成的参数过多，所以论文中使用**Dropout方法隐去一些节点减少计算量**，隐去比例论文给的是0.5。在论文中池化采用MaxPooling，同时激活函数使用Relu，输出层采用softmax。具体核大小和步长可以参见论文3.5节。

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

## VGG16

![VGG16](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/VGG16.png)

### VGG16模型介绍

VGG16来自论文Very Deep Convolutional Networks for Large Scale Image Recognition。论文阐述了6种VGG模型，VGG16是其中一种16层（没把池化算上）的VGG模型。如上图所示，它的结构也很简单：

卷积1 - 卷积2 - 池化1 - 卷积3 - 卷积4 - 池化2 - 卷积5 - 卷积6 - 卷积7 - 池化3 - 卷积8 - 卷积9 - 卷积10 - 池化4 - 卷积11 - 卷积12 - 卷积13 - 池化5 - 全连接1 - 全连接2 - 输出层

输出层为softmax，池化采用maxpooling，激活函数采用Relu。

### VGG16模型实现

按照定义构建即可，注意VGG会耗费更多计算资源，并且使用了更多的参数，导致更多的内存占用，可能会造成**内存溢出**使得程序崩溃，在model.fit的时候应**适当调节batch_size参数**。同时也是因为参数过多，**训练十分缓慢**，建议使用一些预训练模型。

本项目在训练时这个模型进行场景分类时表现并不太好，具体原因待探究。

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

Keras**自带了原版VGG16的代码**，可以引入tensorflow.keras.applications.vgg16直接使用，就是输入输出层的大小要改一下。下面的代码构建的模型和上面的是**等效**的：

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
