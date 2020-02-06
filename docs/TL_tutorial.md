# 深度迁移学习教程

迁移学习，是指利用数据、任务、或模型之间的相似性，将在旧领域学习过的模型，应用于新领域的一种学习过程。

本教程介绍几种深度迁移学习并提供TensorFlow的实现。深度迁移学习是迁移学习下的一个子分类，其他迁移学习的内容可以参考[王晋东的《迁移学习简明手册》](https://github.com/jindongwang/transferlearning-tutorial)。本教程其中对训练的操作可以参考[项目使用教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Use_tutorial.md)。

## 目录

- [Finetune](#Finetune)

## Finetune

![Finetune](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/Finetune.png)

### Finetune介绍

这是最简单的深度迁移学习方法。我们在[模型搭建教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Model_tutorial.md)中已经知道了一个CNN模型是由很多层构成的。论文[*How transferable are features in deep neuralnetworks?*](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)认为神经网络的前几层提取的体征具有普适性，而越往后的层特征越具体。所以可以**保留前几个具有普适性的层不动，而重新训练调整后面几层的权值**，这就是Finetune。

一般Finetune的做法有两种，两种方法大同小异：

1. 冻结预训练模型全部卷积层，只训练自己定制的全连接层。
2. 冻结预训练模型前几个卷积层，训练剩下的卷积层和全连接层。

TIPS： 另外还有一种迁移学习的方式和Finetune类似，它把CNN当作特征提取器，然后将提取到的结果（即CNN训练结果）作为输入重新放入分类器（比如softmax）中训练。

### Finetune实现

先需要利用一个预训练的模型构造一个新的模型，然后冻结新模型前几层，并对后几层进行训练。

#### Finetune模型构建

模型构建参看[模型搭建教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Model_tutorial.md)。由于没有找到合适的Keras场景分类预训练模型，此处使用了Keras自带的预训练模型Inception v3，它用的是[ImageNet](http://www.image-net.org/)进行预训练的。

具体实现的时候要**将原模型的全连接层去掉，并添加上新的全连接层**。在迁移学习的实践中常常使用全局池化层Global Average Pool代替全连接层Dense，全局池化层出自论文[*Network In Network*](https://arxiv.org/pdf/1312.4400.pdf%20http://arxiv.org/abs/1312.4400)，这样效果更好且节约性能。

TIPS: 也可以使用ResNet等其他模型进行Finetune，步骤类似。

```python
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 建立预训练的网络
def build_model(input_shape, output_shape):
    # 为了改输出层得到去掉全连接层的预训练InceptionV3
    model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    # 添加新的全连接层
    tensor = GlobalAveragePooling2D()(model.output)
    tensor = Dense(1024, activation='relu')(tensor)
    tensor = Dense(output_shape, activation='softmax')(tensor)
    model_v3 = Model(inputs=model.input, outputs=tensor, name='inception_v3')
    return model_v3
```

#### Finetune最终实现

下面的代码和之前的[训练模型示例](https://github.com/psiang/Scene_Classification/blob/master/docs/Use_tutorial.md#训练模型示例)的主要区别就在于有没有**冻结操作**。Finetune需要先冻结前几层才能进行训练。

本项目冻结了前172层，用少量数据（训练集:测试集 = 2:8）做训练，且只跑了5个周期，耗费时长仅十多分钟（之前重头训练100个周期要十多个小时），但效果不输于之前任何一个模型，这就是迁移学习的神奇之处。

```python
from rsidea.preprocess import read_data, read_label, split_data
from rsidea.util.draw import *
from rsidea.util.history import *

'''finetune demo'''
# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(x, y, rate=0.8)
# 获取原训练模型
model = build_model(input_shape=x_train[0].shape, output_shape=12)

# 冻结前172层
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 填入数据进行训练
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))
history = history.history
# 模型评测
model.evaluate(x_test, y_test, verbose=2)
# 画折线图
draw_accuracy(history)
```
