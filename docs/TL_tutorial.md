# 深度网络迁移学习教程

迁移学习，是指利用数据、任务、或模型之间的相似性，将在旧领域学习过的模型，应用于新领域的一种学习过程。

本教程介绍几种深度网络迁移学习并提供TensorFlow的实现。深度网络迁移学习是迁移学习下的一个子分类，其他迁移学习的内容可以参考[王晋东的《迁移学习简明手册》](https://github.com/jindongwang/transferlearning-tutorial)。

本教程先介绍最简单的深度网络迁移学习**Finetune**，再以**深度网络自适应**和**深度对抗网络**的两个简单模型DDC、DANN为例子阐述如何进行迁移学习。本教程其中对训练的操作可以参考[项目使用教程](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Use_tutorial.md)。

## 目录

- [Finetune](#Finetune)
- [DDC](#DDC)
- [DANN](#DANN)

## Finetune

![Finetune](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/pics/Finetune.png)

### Finetune介绍

这是最简单的深度迁移学习方法。我们在[模型搭建教程](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Model_tutorial.md)中已经知道了一个CNN模型是由很多层构成的。论文[*How transferable are features in deep neuralnetworks?*](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)认为神经网络的前几层提取的体征具有普适性，而越往后的层学习到的特征越具体。所以可以**保留前几个具有普适性的层不动，而重新训练调整后面几层的权值**，这就是Finetune。

一般Finetune的做法有两种，两种方法大同小异：

1. 冻结预训练模型全部卷积层，只训练自己定制的全连接层。
2. 冻结预训练模型前几个卷积层，训练剩下的卷积层和全连接层。

TIPS： 另外还有一种迁移学习的方式和Finetune类似，它把CNN当作特征提取器，然后将提取到的结果（即CNN训练结果）作为输入重新放入分类器（比如softmax）中训练。

### Finetune实现

先需要利用一个预训练的模型构造一个新的模型，然后冻结新模型前几层，并对后几层进行训练。

- [Finetune模型构建](#Finetune模型构建)
- [Finetune最终实现](#Finetune最终实现)

#### Finetune模型构建

模型构建参看[模型搭建教程](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Model_tutorial.md)。由于没有找到合适的Keras场景分类预训练模型，此处使用了Keras自带的预训练模型Inception v3，它用的是[ImageNet](http://www.image-net.org/)进行预训练的。

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

下面的代码和之前的[训练模型示例](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Use_tutorial.md#训练模型示例)的主要区别就在于有没有**冻结操作**。Finetune需要先冻结前几层才能进行训练。

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

![DDC](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/pics/DDC.png)

### DDC介绍

DDC出自论文[*Deep Domain Confusion: Maximizing for Domain Invariance*](https://arxiv.org/pdf/1412.3474)，是一种深度网络自适应的迁移方法，用来解决Finetune无法处理源数据（迁移前的数据）和目标数据（迁移后的数据）分布不同的情况。

DDC如上图所示在[AlexNet](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Use_tutorial.md#AlexNet)的8层网络基础上，在第7层全连接层之后新加了一个**域适应层**，并**固定前7层的权值**。DDC的输入有两个——源数据和目标数据，源数据有标签而目标数据没有。源数据和目标数据都在**同一个AlexNet上**运行至适应层后，利用源和目的数据在适应层的输出计算**域损失**(*domain loss*)，然后源数据继续跑至分类器层得到预测值，与源数据的标签比较，计算总损失并更新网络。

总损失的计算公式为：

![$$l=l_c(D_s, y_s)+\lambda MMD^2(D_s,D_t)$$](http://latex.codecogs.com/gif.latex?l=l_c(D_s,y_s)+\lambda%20MMD^2(D_s,D_t))

其中***l_c***为预测值***D_s***和真实标签***y_s***之间的损失，这和之前在[使用教程](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Use_tutorial.md#方式一构造模型并训练)中model.complie中的loss参数的含义是一致的；***MMD***是域损失中使用最广泛的一种损失函数，在适应层计算出来。最大均值差异MMD(Maximum Mean Discrepancy)**衡量了两个数据分布的距离**，我们把这个域损失加入损失函数就是为了**缩小数据分布的差距**。

### DDC实现

通过上面的介绍我们了解了DDC和AlexNet有三个区别：

1. 引入了MMD的机制
2. 重新定义了损失函数
3. 模型加入了一个适应层并且有两个输入

下面将一一介绍以上实现。

- [MMD的实现](#MMD的实现)
- [损失函数的实现](#损失函数的实现)
- [DDC模型的实现](#DDC模型的实现)
- [DDC最终实现](#DDC最终实现)

#### MMD的实现

MMD的推导此处不详细叙述, 此处只提供计算公式，可以去看王晋东的手册，他似乎还提供了计算量更小的方法。另外MMD实现看不懂的话**可以当作黑箱先暂时跳过**。

![$$MMD(X,Y)=||\frac{1}{n^2}\sum_i^n\sum_{i'}^n k(x_i,x_{i'})-\frac{2}{nm}\sum_i^n\sum_j^m k(x_i,y_j)+\frac{1}{m^2}\sum_j^n\sum_{j'}^n k(y_j,y_{j'})||$$](http://latex.codecogs.com/gif.latex?MMD(X,Y)=||\\frac{1}{n^2}\\sum_i^n\\sum_{i%27}^n%20k(x_i,x_{i%27})-\\frac{2}{nm}\\sum_i^n\\sum_j^m%20k(x_i,y_j)+\\frac{1}{m^2}\\sum_j^n\\sum_{j%27}^n%20k(y_j,y_{j%27})||)

上式中，核函数***k***为我们在概率论上众所周知的高斯函数：

![$$k(x,y)=e^{\frac{-||x-y||^2}{2\sigma^2}}$$](http://latex.codecogs.com/gif.latex?k(x,y)=e^{\\frac{-||x-y||^2}{2\\sigma^2}})

实现的时候把和式部分用矩阵来表示了，以上式中间项为例，***k***为元素（***x***或***y***）大小，输入是***n × k***维的***X***和***m × k***维的***Y***，求出***n × m***的核矩阵**K**，核矩阵中每一个元素就是***k(x,y)***，对核矩阵求均值Mean就可以算出上式中间那项了。

实现核矩阵的时候用了[数组广播特性](http://cs231n.github.io/python-numpy-tutorial/#numpy-broadcasting)，在第一个向量***X***中间扩展1维变成***n × 1 × k***维再相减，可使得两个向量***X***和***Y***中每个元素两两都做一次相减而不用写循环。这样得到一个***n × m × k***的矩阵，平方后把最后一维相加就可得到***n × m***维的矩阵了，之后再做处理得到核矩阵***K***。大家可以用Numpy实验一下。另外代码把高斯函数的常数部分直接简化用***beta***表示。

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

我们在之前[配置模型](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/Use_tutorial.md#方式一构造模型并训练)的时候使用的是如下代码，loss函数使用了keras自带的sparse categorical crossentropy：

```python
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

这相当于只包括了新定义损失的l_c的部分。为了实现新定义的总损失，我们需要自己**重新构造一个损失函数**。

Keras允许loss的参数是一个[自定义损失函数](https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618)，但Keras的损失函数规定只能有两个参数y_true和y_pred——即真实值***y_s***和预测值***D_s***。现在为了将MMD的值代入计算需要额外多一个参数。我们利用[python返回函数的特性](https://www.liaoxuefeng.com/wiki/897692888725344/989705420143968)，函数中套一个函数，可以解决需要更多参数的问题。公式中的***lambda***的值在论文中设为0.25。

```python
from tensorflow.keras.losses import sparse_categorical_crossentropy

# ddc损失函数
def loss_ddc(mmd):
    def loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred) + 0.25 * mmd * mmd
    return loss
```

#### DDC模型的实现

解决多输入问题。需要建立两个输入层。由于源数据和目的数据要在同一个AlexNet上跑，所以这里要用到keras的[权值共享网络](https://keras.io/zh/getting-started/functional-api-guide/)。具体操作就是先构建一个网络，**将其用Model只实例化一次**，对实例重复使用就相当于在同一个网络上跑。此外**只用在训练更新网络时跑目的数据**，测试时只需要得到预测值所以不用跑了，这样可以节约测试时间，用backend中的learning_phase可判断是否处于训练阶段。

解决适应层问题。需要把AlexNet的第七层的输出，接在一个新的全连接层上，这个新的全连接层就是适应层。**使用get_layer函数**可得到模型中某一层。适应层的维数在论文中定位256。

以下便是实现代码。它先构建了**预训练并冻结了前七层且带适应层AlexNet**，然后实例化这个AlexNet。再在DDC模型构建中定义两个输入，两个输入分别穿过AlexNet实例，计算MMD，把源数据代入分类器。最后构建并返回DDC模型以及计算好的mmd。

```python
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

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
    # 在AlexNet上跑源数据
    tensor_1 = model_alex(inputs_1)
    # 在训练的时候才在AlexNet上跑目的数据
    mmd = 0
    if K.learning_phase() == 1:
        # 在AlexNet上跑目的数据
        tensor_2 = model_alex(inputs_2)
        # 计算mmd
        mmd = loss_mmd(tensor_1, tensor_2)
    # 源数据进入分类器
    tensor = Dense(output_shape, activation='softmax')(tensor_1)
    model = Model(inputs=[inputs_1, inputs_2], outputs=tensor, name='ddc')
    return model, mmd
```

#### DDC最终实现

模型都建立好了，其他部分就简单了，和之前的模型没有太大区别。需要注意三点：一是模型构建返回了mmd；二是配置模型的时候损失函数要**改成DDC的损失函数**，并代入mmd；三是有模型两个输入，训练时应代入源和目的数据，模型评估和预测的时候可以两个输入都是目的数据。

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
# 模型预测，输入都用目的数据
prediction = model.predict([target, target])
```

## DANN

![DANN](https://github.com/psiang/TensorFlow-Deep_Learning_and_Transfer_Learning/blob/master/docs/pics/DANN.png)

### DANN介绍

DANN出自论文[*Domain-Adversarial Training of Neural Networks*](https://dl.acm.org/doi/pdf/10.5555/2946645.2946704?download=true),是一种深度对抗网络的迁移方法。设定源数据为迁移前的数据，目的数据为迁移后的数据，源数据有标签而目的数据没有。

DANN的结构如上图所示，被分成了三大块：**特征提取器(*feature extractor*)、预测器(*label predictor*)、判别器(*domain classifier*)**。其中特征提取器和预测器组合起来与之前的神经网络的结构没有区别——特征提取器就是卷积层、预测器是全连接层和输出层。判别器是能够区分源数据和目的数据的全连接层网络。但我们迁移学习的目的是使得特征网络不能区分源数据和目的数据，从而消除两个域之间的差距，所以**判别器在梯度下降传到特征提取器的时候应该反号**，变成“梯度上升”。这样来自预测器的梯度下降和来自判别器的“梯度上升”形成了对抗。

为此论文在特征提取器和判别器之间加入了**梯度反向层GRL(*Gradient Reversal Layer*)**，GRL该层前向传播时不进行任何变化，后向传播即梯度下降时将梯度反号处理。DANN的损失函数和DDC类似，将判别器的损失作为了补充：

![$$l=l_c(D_s, y_s)+\lambda l_d^2(D_s,D_t)$$](http://latex.codecogs.com/gif.latex?l=l_c(D_s,y_s)+\lambda%20l_d(D_s,D_t))

### DANN实现

论文提供了三种架构。项目采用了以LeNet为基础，加上了GRL和判别器的实现。在论文实现的时候，**每一批的训练源数据和目的数据各占一半**，并且还要给源和目的数据加上判别标签，所以还需要对数据进行预处理。论文的模型还设置了**动态学习率**。下面将先介绍GRL和模型的实现，然后介绍数据预处理和动态学习率的实现，最后介绍最终的模型使用实现。

- [梯度反向层GRL的实现](#梯度反向层GRL的实现)
- [DANN模型实现](#DANN模型实现)
- [DANN预处理实现](#DANN预处理实现)
- [动态学习率实现](#动态学习率实现)
- [DANN最终实现](#DANN最终实现)

#### 梯度反向层GRL的实现

首先需要继承Layer**重构一个新的层**。call函数需要重载用于tensor的处理，其输入参数x即为来自上一层的tensor输入，其return返回的参数即为通过该层处理后给下一层的输出tensor。其他函数重载不多做介绍，参见[TensorFlow自定义层](https://www.tensorflow.org/guide/keras/custom_layers_and_models)。

其次需要[自定义GRL梯度](https://stackoverflow.com/questions/52084911/how-to-create-a-custom-layer-to-get-and-manipulate-gradients-in-keras)。从之前的实践中我们知道TensorFlow是先构建出图，然后再填入数据运行的，为了修改梯度我们要修改图。如下所示使用修饰器RegisterGradient**注册一个新梯度函数**，函数中让梯度反号。然后再获取TensorFlow的图，用gradient_override_map**替换Identity的梯度**为自己新注册的梯度，Identity的作用就是返回跟输入完全相同的输出。

最后梯度的名称是不能重复的，为了防止被多次调用，设置一个计数器num_calls对梯度的名称在每次调用时进行修改。

```python
import tensorflow as tf
from tensorflow.keras.layers import Layer


# 构建GRL
class GradientReversal(Layer):
    def __init__(self, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.num_calls = 0

    # 该层处理张量
    def call(self, x, mask=None):
        # 设置不重复的梯度名称
        grad_name = "GradientReversal%d" % self.num_calls
        # 注册反号的梯度
        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * 1.0]
        # 获取当前会话的图
        g = tf.compat.v1.get_default_graph()
        # 替换identity的梯度为自己新定义的梯度
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(x)

        self.num_calls += 1

        return y
```

#### DANN模型实现

模型有特征提取器、预测器和判别器三个重要的组成部分，下面先介绍这三个部分，再说明这三个部分如何连接在一起。

##### 特征提取器

特征提取器按照LeNet卷积部分直接构建即可：

```python
# 特征提取器（使用LeNet）
def __feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    tensor = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(inputs)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)

    tensor = Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(tensor)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)

    tensor = Dropout(0.5)(tensor)
    tensor = Flatten()(tensor)

    feature_output = Dense(100, activation="relu")(tensor)
    # 实例化提取器
    model = Model(inputs=inputs, outputs=feature_output, name='feature_extractor')
    return model
```

##### 预测器

预测器按照LeNet全连接部分直接构建即可：

```python
# 预测器
def __label_predictor(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    out = Dense(128, activation="relu")(inputs)
    out = Dropout(0.5)(out)
    predictor_output = Dense(output_shape, activation="softmax", name="classifier_output")(out)
    # 实例化预测器
    model = Model(inputs=inputs, outputs=predictor_output, name='label_predictor')
    return model
```

##### 判别器

判别器开始先插入一个梯度反向层GRL，然后接一个全连接层，最后时一个输出层，注意输出的类别只有2种（即判别是源还是目的），是二分类。

```python
# 判别器
def __domain_classifier(input_shape):
    inputs = Input(shape=input_shape)
    # 插入GRL
    grl_layer = GradientReversal()
    out = grl_layer(inputs)
    out = Dense(128, activation="relu")(out)
    out = Dropout(0.5)(out)
    classifier_output = Dense(2, activation="softmax", name="discriminator_output")(out)
    # 实例化判别器
    model = Model(inputs=inputs, outputs=classifier_output, name='domain_classifier')
    return model
```

##### 组成DANN模型

在前面介绍到，训练的时候，每一批数据是一半源数据和一半目的数据组成，这批数据通过特征提取器后，可以直接给判别器；但是由于目的数据是没有真实标签的，所以**只能把源数据给预测器**，目的数据应丢弃。

利用Lambda可以快速构建一层，下面的代码在Lambda层，先用learning_phase判断是否在训练阶段，用switch实现分支结构。如果当前是训练阶段，则将这一批的前一半（即源数据）复制拼接，即使得改批从 **源-目的** 变成 **源-源**；如果是测试阶段则不作处理。返回数据的大小和输入相比不变。

TIPS：此处也可以像GRL一样继承Layer构建一个自定义层，不够没有这个简单方便。

```python
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model

from rsidea.util.layers import GradientReversal


# DANN建模
def DANN(input_shape, output_shape, batch_size=32):
    # 输入
    inputs = Input(shape=input_shape, name="source_input")
    # 通过特征提取器
    feature_extractor = __feature_extractor(input_shape=inputs[0].shape)
    feature_output = feature_extractor(inputs)
    # 通过判别器
    domain_classifier = __domain_classifier(input_shape=feature_output[0].shape)
    domain_classifier_output = domain_classifier(feature_output)
    # 处理数据
    source_feature = Lambda(lambda x: K.switch(K.learning_phase(),
                                               K.concatenate([x[:int(batch_size // 2)], x[:int(batch_size // 2)]],
                                                             axis=0),
                                               x),
                            output_shape=lambda x: x)(feature_output)
    # 通过分类器
    label_predictor = __label_predictor(input_shape=feature_output[0].shape, output_shape=output_shape)
    label_predictor_output = label_predictor(source_feature)
    # 实例化模型
    model = Model(inputs=inputs, outputs=[label_predictor_output, domain_classifier_output])
    return model
```

#### DANN预处理实现

我们应将每一批数据处理成一半源数据和一半目的数据。同时按照[模型构建](#组成DANN模型)，源数据的标签也应该**每半批复制拼接成新的一批**。还应该增加域判别标签，**每批由半批0和半批1拼接而成**，表示源和目的数据输于不同的域类别。

```python
import numpy as np

# 数据预处理
def preprocess_data(x_source, y_source, x_target, batch_size=32):
    # 位置计数，半批一处理
    index = 0
    # 半批
    half_batch = int(batch_size // 2)
    # 表示处理后的图像数据、判别标签和源数据标签
    images = []
    domains = []
    truths = []
    while x_source.shape[0] > index + half_batch:
        # 图像数据：一半源数据和一半目的数据
        batch_images = np.concatenate((x_source[index: index + half_batch],
                                       x_target[index: index + half_batch]), axis=0)
        # 判别标签：每批由半批0和半批1拼接而成
        batch_domains = np.concatenate((np.array([0] * half_batch),
                                        np.array([1] * half_batch)), axis=0)
        # 源数据标签：每半批复制拼接成新的一批
        batch_truths = np.concatenate((y_source[index: index + half_batch],
                                       y_source[index: index + half_batch]), axis=0)
        # 每批整合到一起
        if index == 0:
            images = batch_images
            domains = batch_domains
            truths = batch_truths
        else:
            images = np.concatenate((images, batch_images), axis=0)
            domains = np.concatenate((domains, batch_domains), axis=0)
            truths = np.concatenate((truths, batch_truths), axis=0)
        index += half_batch
    return images, truths, domains
```

#### 动态学习率实现

论文在实现的时候使用了**动态学习率μ**和**动态预适应参数λ**，即这两个参数随着epoch的周期不断变化：

![$$\mu_p=\frac{\mu_0}{(1+\alpha p)^\beta}$$](http://latex.codecogs.com/gif.latex?\\mu_p=\\frac{\\mu_0}{(1+\\alpha%20p)^\\beta})

![$$\lambda_p=\frac{2}{1+\exp(-\gamma p)}-1$$](http://latex.codecogs.com/gif.latex?\lambda_p=\frac{2}{1+\exp(-\gamma%20p)}-1)

其中p是随着epoch在0到1上线性变化的参数，α、β、γ、μ_0是固定参数，论文中提供了具体数值。μ就是整个模型的学习率，λ是总损失中对应着域适应的权值。

本项目只实现了μ的动态变化，而固定了λ为0.31。实现μ的动态变化需要重构LearningRateScheduler中的scheduler函数，并将新的LearningRateScheduler作为callbacks在模型训练的时候传入。这个scheduler函数将在每个epoch开始的时候调用确认新的学习率。

```python
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K


def lr_dann(model, epochs):
    # 按照论文公式设置学习率
    def scheduler(epoch):
        u0 = 0.01
        a = 10.0
        b = 0.75
        p = epoch / (epochs * 1.0)
        lr = u0 / ((1 + a * p) ** b)
        K.set_value(model.optimizer.lr, lr * 0.1)
        return K.get_value(model.optimizer.lr)
    # 形成处理学习率的Callback
    reduce_lr = LearningRateScheduler(scheduler)
    return reduce_lr
```

#### DANN最终实现

和之前的最终实现有以下区别：

1. 读取数据后进行了对批的预处理。
2. **模型为双输出**，所以在compile的时候对每个输出都应该指定损失函数，fit的时候应该把源数据标签和判别标签代入。
3. 模型训练的时候要将调整学习率的callbacks传入
4. 预测应该用数据预处理前的目的数据。

TIPS：DANN模型也可以导入预训练的权值再对特征提取器进行Finetune。

```python
from rsidea.models import *
from rsidea.preprocess import read_data, read_label, split_data
from rsidea.util.callbacks import lr_dann

BATCH_SIZE = 32
EPOCH = 100

'''dann demo'''
# 读取数据
x_source, y_source, x_target = ...
# 数据预处理
images, truths, domains = dann.preprocess_data(x_source, y_source, x_target, batch_size=BATCH_SIZE)
# 获取原训练模型
model = dann.DANN(input_shape=x_source[0].shape, output_shape=12, batch_size=BATCH_SIZE)
model.summary()
# 配置模型
model.compile(optimizer='adam',
              loss={'label_predictor': 'sparse_categorical_crossentropy',
                    'domain_classifier': 'sparse_categorical_crossentropy'},
              loss_weights={'label_predictor': 1.0,
                            'domain_classifier': 0.31},
              metrics=['accuracy'],
              experimental_run_tf_function=False)
# 填入数据进行训练
history = model.fit(images, [truths, domains], epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr_dann(model, EPOCH)])
history = history.history
# 模型预测
prediction = model.predict(x_target)
```
