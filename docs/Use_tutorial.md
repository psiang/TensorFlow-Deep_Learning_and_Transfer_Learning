# 项目使用教程

本教程按照以下五步展示如何使用本项目进行场景分类，并提供tensorflow的部分使用方法：

1. 数据的读取
2. 数据的预处理
3. 模型获取和保存
4. 模型测试
5. 结果图像生成

最后将提供两个使用示例代码提供参考。

另外，本教程所有操作均在__main__.py完成。教程只提供项目自己的接口的具体参数介绍，关于tensorflow的接口参数和功能可查看[官网文档](https://www.tensorflow.org/api_docs/python/tf)。而构建CNN模型的部分则转移至另一个[模型搭建教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Model_tutorial.md)。

参考文献：[tensorflow官网CNN教程](https://www.tensorflow.org/tutorials/images/cnn)。

## 目录

- [引入模块](#引入模块)
- [数据的读取](#数据的读取)
  - [样本数据读取](#样本数据读取)
  - [标签名称获取](#标签名称获取)
- [数据的预处理](#数据的预处理)
  - [分割数据](#分割数据)
- [模型获取和保存](#模型获取和保存)
  - [获取模型](#获取模型)
  - [保存模型](#保存模型)
  - [训练历史的存取](#训练历史的存取)
- [模型测试](#模型测试)
  - [模型评估](#模型评估)
  - [模型预测](#模型预测)
- [结果图像生成](#结果图像生成)
  - [损失折线图](#损失折线图)
  - [准确度折线图](#准确度折线图)
  - [单张预测展示图](#单张预测展示图)
- [示例代码](#示例代码)
  - [训练模型示例](#训练模型示例)
  - [加载模型示例](#加载模型示例)

## 引入模块

只需要引入rsidea.model即可，其他的模块都已经在这个模块下引入。

```python
from rsidea.model import *
```

## 数据的读取

### 样本数据读取

```python
# 读取数据
x, y = read_data.read_SIRI_WHU()
```

**Arguments:**

- data_dir：字符串类型。数据文件夹位置，默认为".\\data\\Google dataset of SIRI-WHU_earth_im_tiff\\12class_tif"，可手动提供。

**Return:**

- x：Numpy类型。图像样本数据，有四维，第1维是不同的图像，第2至3维是图像的像素长宽，第4维是图像通道数。**注意学习器的数据必须是0~1之间的实数。**
- y：Numpy类型。样本的标签，返回的是编号过的标签。**注意学习器的标签必须是数字，此处若自己写接口必须将标签转换成int类型。**

TIPS：本项目只提供了对样例数据SIRI_WHU的读入接口，如果需要对其他数据进行读取则按照类似[规范代码](https://github.com/psiang/Scene_Classification/blob/master/rsidea/preprocess/read_data.py)补充接口即可。

### 标签名称获取

```python
# 读取标签名称
names = read_label.read_SIRI_WHU()
```

**Arguments:**

- data_dir：字符串类型。数据文件夹位置，默认为".\\data\\Google dataset of SIRI-WHU_earth_im_tiff\\12class_tif"，可手动提供。

**Return:**

- names：Numpy类型。样本的数据，如果要获取原始标签的名称则通过这个接口获取，是原始字符串形式的标签。

TIPS：本项目只提供了对样例数据SIRI_WHU的读入接口，如果需要对其他数据标签进行读取则按照类似[规范代码](https://github.com/psiang/Scene_Classification/blob/master/rsidea/preprocess/read_label.py)补充接口即可。

## 数据的预处理

暂时只提供了分割数据这一种预处理。

### 分割数据

```python
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(data, label)
```

**Arguments:**

- data：Numpy类型。需要分割的样本数据。
- label：Numpy类型。需要分割的样本标签。
- rate：float类型。分割出用作测试集的比例，默认为0.2，可手动设置。

**Return:**

- x_train：Numpy类型。训练数据。
- y_train：Numpy类型。训练数据对应的标签。
- x_test：Numpy类型。测试数据。
- y_test：Numpy类型。测试数据对应的标签。

## 模型获取和保存

这一小节专注于效果实现，models相关的接口请参考[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/Model)。

一个tensorflow的models分为**模型构成的图**和**图上的权值**两个部分。一个构造模型在未训练的时候上面是没有权值的，只有图本身；而模型训练以后图上就有相应的权值了。模型和权值可以**分别存取**。

### 获取模型

通过以下的方式可以得到已训练的模型：

- 构造模型并训练
- 构造模型加载预训练权值
- 载预训练模型

构造模型可以参考另一篇[模型搭建教程](https://github.com/psiang/Scene_Classification/blob/master/docs/Model_tutorial.md)，调用构造的接口可以获得一个Model类型的未训练模型。

#### 方式一:构造模型并训练

当没有预训练时只能采取这种方法得到训练模型。主要有三步：

1. 获取构建的模型
2. 配置模型
3. 装填数据进行训练

如果有绘图需要可以如下面的代码所示获取训练历史数据。

```python
# 获取未训练模型
model = LeNet(input_shape=x_train[0].shape, output_shape=12)
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 填入数据训练
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
# 获取训练历史
history = history.history
```

#### 方式二:构造模型加载预训练权值

当预训练并保存了图的权值时可以使用这个方法。主要有三步：

1. 获取构建的模型
2. 配置模型
3. 加载权值数据

需要注意的是*构造的模型*必须和*权值数据训练时的模型***相同**才可以被正确加载。

```python
# 获取未训练模型
model = LeNet(input_shape=x_train[0].shape, output_shape=12)
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 加载权值数据
model.load_weights(".\\model_data\\weight\\lenet_SIRI_WHU.h5")
```

#### 方式三:加载预训练模型

当预训练并保存了模型时可以使用这个方法。直接调用接口加载即可，注意调用的是models模块函数而不是实例的函数。

```python
# 加载模型
model = models.load_model(".\\model_data\\model\\lenet_SIRI_WHU.h5")
```

### 保存模型

可以用两种方式保存模型，要么把整个模型直接保存下来，然后用[方式三](#方式三载预训练模型)加载模型；要么只保存模型的权值，然后用[方式二](#方式二构造模型加载预训练权值)加载模型。

保存模型请用.**h5**的形式

#### 保存整个模型

把整个模型直接保存下来，可用[方式三](#方式三加载预训练模型)加载模型。这样做的好处是加载简单方便且可方便移植。

```python
# 保存模型
model.save(".\\model_data\\model\\lenet_SIRI_WHU.h5")
```

#### 仅保存模型权值

只保存模型的权值，可用[方式二](#方式二构造模型加载预训练权值)加载模型。这样做的好处是比保存整个模型所需空间更小。

```python
# 保存模型权值
model.save_weights(".\\model_data\\weight\\lenet_SIRI_WHU.h5")
```

### 训练历史的存取

在获取模型的[方式一](#方式一构造模型并训练)中训练数据可以返回一个训练历史数据，历史数据与配置模型时metrics参数有关。这个历史数据可以用于[生成图像](#结果图像生成)进行分析，所以我们利用json把它保存下来。

TIPS：model.fit返回值的是一个History类型的实例，我们所需要的历史数据在History.history中，它是一个字典类型数据。

#### 保存训练历史

```python
# 保存训练历史
save_history(history, ".\\model_data\\history\\lenet_SIRI_WHU.json")
```

**Arguments:**

- history：字典类型。训练的历史数据。
- path：字符串类型。数据保存的位置。

#### 加载训练历史

```python
# 加载训练历史
history = load_history(".\\model_data\\history\\lenet_SIRI_WHU.json")
```

**Arguments:**

- path：字符串类型。训练历史数据保存的位置。

**Return:**

- history：字典类型。训练历史数据。

## 模型测试

### 模型评估

带入测试数据即可，返回值与配置模型时metrics参数有关，具体见[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/Model)。

```python
# 模型评估
loss, acc = model.evaluate(x_test, y_test, verbose=2)
```

### 模型预测

支持测试一组数据，所以如果预测单个数据必须扩展成跟训练数据一样的四维，返回值为这组数据的预测结果，每个结果是输出层的值（输出层用softmax即为该样本是该标签的概率），具体见[官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/Model)。

```python
# 模型预测
y = model.predict(x)
```

## 结果图像生成

### 损失折线图

history中一定有每一轮的loss数据，可生成折线图。接口内暂时未设置保存图像代码，需要手动保存。

```python
# 画损失折线图
draw_loss(history)
```

**Arguments:**

- history：字典类型。训练历史数据。

图例：

![损失折线图](https://github.com/psiang/Scene_Classification/blob/master/results/loss/lenet_SIRI_WHU.png)

### 准确度折线图

配置模型时metrics参数中加入'accuracy'则在history中可得到准确度的每一轮历史数据，也可以画折线图。

```python
# 画准确度折线图
draw_accuracy(history)
```

**Arguments:**

- history：字典类型。训练历史数据。

图例：

![准确度折线图](https://github.com/psiang/Scene_Classification/blob/master/results/accuracy/lenet_SIRI_WHU.png)

### 单张预测展示图

单张预测展示图可方便地用于学习展示，展示图最多只展示5个可能性最大的标签。示例所测试的图demo.jpg是从网上的一张卫星图上截取的。

先通过以下方式[获取标签名称](#标签名称获取)和图像，注意输入。

```python
# 获取标签
label_name = read_label.read_SIRI_WHU()
# 读取demo图像数据
x = mpimg.imread(".\\data\\demo.jpg") / 255.0
```

再画单张预测图像。

```python
# 画单张预测展示图
draw_predict_demo(model, x, label_name)
```

**Arguments:**

- model：Model类型。训练好的模型。
- x：Numpy类型。单张图片样本数据，由于只有一张图片，所以只有三维，即像素长宽和通道数。**样本数据要转换成0~1之间的实数。**
- label_name：Numpy类型。对应的标签名称列表。
  
图例：

![单张预测展示图](https://github.com/psiang/Scene_Classification/blob/master/results/prediction/lenet_SIRI_WHU.png)

## 示例代码

### 训练模型示例

以AlexNet为例的训练示例代码。

```python
from rsidea.model import *

# 用于控制是否保存模型
save = False

"""AlexNet demo"""
# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(x, y)
# 获取未训练模型
model = AlexNet(input_shape=x_train[0].shape, output_shape=12)
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 填入数据进行训练
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))
history = history.history
# 模型保存
if save:
    model.save(".\\model_data\\model\\alexnet_SIRI_WHU.h5")
    model.save_weights(".\\model_data\\weight\\alexnet_SIRI_WHU.h5")
    save_history(history, ".\\model_data\\history\\alexnet_SIRI_WHU.json")
    print("Saved!")
# 模型评测
model.evaluate(x_test, y_test, verbose=2)
# 画准确度折线图
draw_accuracy(history)
```

### 加载模型示例

以LeNet为例的加载模型示例代码。

```python
from rsidea.model import *

"""LeNet demo"""
# 加载模型
model = models.load_model(".\\model_data\\model\\lenet_SIRI_WHU.h5")
# 获取标签
label_name = read_label.read_SIRI_WHU()
# 读取demo图像数据
x = mpimg.imread(".\\data\\demo.jpg") / 255.0
# 画单张预测展示图
draw_predict_demo(model, x, label_name)
```
