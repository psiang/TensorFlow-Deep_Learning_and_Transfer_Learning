# 深度网络迁移学习教程

迁移学习，是指利用数据、任务、或模型之间的相似性，将在旧领域学习过的模型，应用于新领域的一种学习过程。

本教程介绍几种深度网络迁移学习并提供TensorFlow的实现。深度网络迁移学习是迁移学习下的一个子分类，其他迁移学习的内容可以参考[王晋东的《迁移学习简明手册》](https://github.com/jindongwang/transferlearning-tutorial)。

本教程先介绍最简单的深度网络迁移学习**Finetune**，再举例说明**深度网络自适应**和**深度对抗网络**如何进行迁移学习。本教程其中对训练的操作可以参考[项目使用教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Use_tutorial.md)。

## 目录

- [Finetune](#Finetune)
- [DDC](#DDC)

## Finetune

![Finetune](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/Finetune.png)

### Finetune介绍

这是最简单的深度迁移学习方法。我们在[模型搭建教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Model_tutorial.md)中已经知道了一个CNN模型是由很多层构成的。论文[*How transferable are features in deep neuralnetworks?*](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)认为神经网络的前几层提取的体征具有普适性，而越往后的层学习到的特征越具体。所以可以**保留前几个具有普适性的层不动，而重新训练调整后面几层的权值**，这就是Finetune。

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
def InceptionV3(input_shape, output_shape):
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
from rsidea.models import *
from rsidea.preprocess import read_data, read_label, split_data
from rsidea.util.draw import *
from rsidea.util.history import *

'''finetune demo'''
# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(x, y, rate=0.8)
# 获取原训练模型
model = inception_v3.InceptionV3(input_shape=x_train[0].shape, output_shape=12)

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

## DDC

![DDC](https://github.com/psiang/Scene_Classification/blob/master/docs/pics/DDC.png)

### DDC介绍

DDC出自论文[*Deep Domain Confusion: Maximizing for Domain Invariance*](https://arxiv.org/pdf/1412.3474)，是一种深度网络自适应的迁移方法，用来解决Finetune无法处理源数据（迁移前的数据）和目标数据（迁移后的数据）分布不同的情况。

DDC如上图所示在[AlexNet](https://github.com/psiang/Scene_Classification/blob/master/docs/Use_tutorial.md#AlexNet)的8层网络基础上，在第7层全连接层之后新加了一个**域适应层**，并**固定前7层的权值**。DDC的输入有两个——源数据和目标数据，源数据有标签而目标数据没有。源数据和目标数据都在**同一个AlexNet上**运行至适应层后，利用源和目的数据在适应层的输出计算**域损失**(*domain loss*)，然后源数据继续跑至分类器层得到预测值，与源数据的标签比较，计算总损失并更新网络。

总损失的计算公式为：

![$$l=l_c(D_s, y_s)+\lambda MMD^2(D_s, D_t)$$](http://latex.codecogs.com/gif.latex?l=l_c(D_s, y_s)+\\lambda MMD^2(D_s, D_t))

其中$l_c(D_s, y_s)$为预测值$D_s$和真实标签$y_s$之间的损失，这和之前在[使用教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Use_tutorial.md#方式一构造模型并训练)中model.complie中的loss参数的含义是一致的；$MMD$是域损失中使用最广泛的一种损失函数，在适应层计算出来。最大均值差异MMD(Maximum Mean Discrepancy)**衡量了两个数据分布的距离**，我们把这个域损失加入损失函数就是为了**缩小数据分布的差距**。

### DDC实现

通过上面的介绍我们了解了DDC和AlexNet有三个区别：

1. 引入了MMD的机制
2. 重新定义了损失函数
3. 模型加入了一个适应层并且有两个输入

下面将一一介绍以上实现。

#### MMD的实现

MMD的推导此处不详细叙述, 此处只提供计算公式，可以去看王晋东的手册，他似乎还提供了计算量更小的方法。另外MMD实现看不懂的话**可以当作黑箱先暂时跳过**。
$$MMD(X,Y)=||\frac{1}{n^2}\sum_i^n\sum_{i'}^n k(x_i,x_{i'})-\frac{2}{nm}\sum_i^n\sum_j^m k(x_i,y_j)+\frac{1}{m^2}\sum_j^n\sum_{j'}^n k(y_j,y_{j'})||$$
上式中，核函数$k$为我们在概率论上众所周知的高斯函数：
$$k(x,y)=e^{\frac{-||x-y||^2}{2\sigma^2}}$$

实现的时候把和式部分用矩阵来表示了，以其中一个为例，$k$为元素（$x_i$或$y_j$）大小，输入是$n\times k$维的$X$和$m\times k$维的$Y$，求出$n \times m$的核矩阵$K_{x,y}$，核矩阵中每一个元素就是$k(x_i,y_j)$，对核矩阵求均值Mean就可以算出$\frac{1}{nm}\sum_i^n\sum_j^m k(x_i,y_j)$。

实现核矩阵的时候用了[数组广播特性](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting)，在第一个向量$X$中间扩展1维变成$n\times 1 \times k$维再相减，可使得两个向量$X$和$Y$中每个元素两两都做一次相减而不用写循环。这样得到一个$n\times m \times k$的矩阵，平方后把最后一维相加就可得到$n \times m$维的矩阵了，之后再做处理得到核矩阵$K_{x,y}$。大家可以用Numpy实验一下。另外代码把高斯函数的常数部分直接简化用beta表示。

```python
import tensorflow as tf
from tensorflow.keras import backend as K


# MMD损失计算
def loss_mmd(x, y):
    xx = __gaussian_kernel(x, x)
    xy = __gaussian_kernel(x, y)
    yy = __gaussian_kernel(y, y)
    loss = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
    return loss


# 高斯核函数的计算
def __gaussian_kernel(x1, x2, beta=1.0):
    # 中间扩展1维
    r = tf.expand_dims(x1, 1)
    # 得到一个n*m的矩阵，为高斯函数的幂
    power = -beta * K.sum(K.square(r - x2), axis=-1)
    return K.exp(power)
```

#### 损失函数的实现

我们在之前[配置模型](https://github.com/psiang/Scene_Classification/blob/master/docs/Use_tutorial.md#方式一构造模型并训练)的时候使用的是如下代码，loss函数使用了keras自带的sparse categorical crossentropy：

```python
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

这相当于只包括了$l_c(D_s, y_s)$的部分。为了实现新定义的总损失，我们需要自己**重新构造一个损失函数**。

Keras允许loss的参数是一个[自定义损失函数](https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618)，但Keras的损失函数规定只能有两个参数y_true和y_pred——即真实值$y_s$和预测值$D_s$。现在为了将MMD的值代入计算需要额外多一个参数。我们利用[python返回函数的特性](https://www.liaoxuefeng.com/wiki/897692888725344/989705420143968)，函数中套一个函数，可以解决需要更多参数的问题。公式中的$\lambda$在论文中为0.25。

```python
from tensorflow.keras.losses import sparse_categorical_crossentropy

# ddc损失函数
def loss_ddc(mmd):
    def loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred) + 0.25 * mmd * mmd
    return loss
```

#### DDC模型的实现

解决多输入问题。需要建立两个输入层。由于源数据和目的数据要在同一个AlexNet上跑，所以这里要用到keras的[权值共享网络](https://keras.io/zh/getting-started/functional-api-guide/)。具体操作就是先构建一个网络，**将其用Model只实例化一次**，对实例重复使用就相当于在同一个网络上跑。

解决适应层问题。需要把AlexNet的第七层的输出，接在一个新的全连接层上，这个新的全连接层就是适应层。**使用get_layer函数**可得到模型中某一层。适应层的维数在论文中定位256。

以下便是实现代码。它先构建了**预训练并冻结了前七层且带适应层AlexNet**，然后实例化这个AlexNet。再在DDC模型构建中定义两个输入，两个输入分别穿过AlexNet实例，计算MMD，把源数据代入分类器。最后构建并返回DDC模型以及计算好的mmd。

```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from rsidea.models import alexnet
from rsidea.util.losses import loss_mmd


# DDC建模
def DDC(input_shape, output_shape):
    """构建带适应层的AlexNet"""
    # 先读入AlexNet模型权值
    model_alex = alexnet.AlexNet(input_shape=input_shape, output_shape=output_shape)
    model_alex.load_weights(".\\model_data\\weight\\alexnet_SIRI_WHU.h5")
    # 冻结前7层卷积和全连接，这里写12是因为中间有池化等
    for layer in model_alex.layers[:12]:
        layer.trainable = False
    for layer in model_alex.layers[12:]:
        layer.trainable = True
    # 获取第七层的输出
    tensor = model_alex.get_layer('dense_1').output
    # 添加适应层
    tensor = Dense(256, activation='relu')(tensor)
    # 实例化带适应层的AlexNet
    model_alex = Model(inputs=model_alex.input, outputs=tensor, name='alexnet')
    """构建DDC模型"""
    # 两个输入分别为源数据和目的数据
    inputs_1 = Input(shape=input_shape)
    inputs_2 = Input(shape=input_shape)
    # 两个输入在同一个AlexNet上跑
    tensor_1 = model_alex(inputs_1)
    tensor_2 = model_alex(inputs_2)
    # 计算mmd
    mmd = loss_mmd(tensor_1, tensor_2)
    # 源数据进入分类器
    tensor = Dense(output_shape, activation='softmax')(tensor_1)
    # 构建双输入模型单输出模型
    model = Model(inputs=[inputs_1, inputs_2], outputs=tensor, name='ddc')
    return model, mmd
```

#### DDC最终实现

模型都建立好了，其他部分就简单了，和之前的模型没有太大区别。需要注意两点，一是模型构建返回了mmd，二是配置模型的时候损失函数要**改成DDC的损失函数**，并代入mmd。

TIPS：由于本项目没有找到合适的预训练AlexNet，所以并没有实际跑DDC。如果找到了合适的，在数据处理时**应使得源数据和目的数据的大小和数目都相同**。

```python
from rsidea.models import *
from rsidea.preprocess import read_data, split_data
from rsidea.util.losses import loss_ddc

'''ddc demo'''
# 获取源数据、源真实值和目标数据
source, source_label, target = ...
# 获取原训练模型
model, mmd = ddc.DDC(input_shape=source[0].shape, output_shape=12)
# 配置模型
model.compile(optimizer='adam',
              loss=loss_ddc(mmd),
              metrics=['accuracy'],
              experimental_run_tf_function=False)
# 填入数据进行训练
history = model.fit([source, target], source_label, epochs=5)
history = history.history
```
