# 模型搭建教程

本教程将简要介绍几种经典的CNN模型，并提供在tensorflow2.0中的实现方式，大家可以以此为例实现其他CNN模型的构建。

tensorflow2.0中已经内置了keras，keras可以方便地将模型构建出来。Model和Layer类型的具体操作参阅[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers)，此处不赘述。所有的模型搭建函数将返回一个搭建好的Model类型。

参考文献：[keras实现常用深度学习模型](https://blog.csdn.net/wmy199216/article/details/71171401)。

## 目录

- [LeNet](#LeNet)
- [AlexNet](#AlexNet)

## LeNet

![LeNet](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/LeNet.png)

如上图所示，LeNet的结构比较简单，即卷积1 - 池化1 - 卷积2 - 池化2 - 全连接1 - 全连接2 - softmax输出层。

在这里，池化我们采用MaxPooling，同时激活函数使用Relu，并简化全连接层为一层，由此可以构建以下代码。

TIPS：如果需要复现LeNet5，只需添加一层全连接1即可。另外[有资料认为](https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b)（比如上图）全连接1应该做卷积操作而不是传统意义上的全连接。

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

