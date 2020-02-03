from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import vgg16


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


# 带BN层的卷积
def __Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        bn_name = None
        conv_name = None
        act_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Activation('relu', name=act_name)(x)
    return x


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
