import os
import tensorflow as tf
from rsidea.util.history import *
from rsidea.util.draw import *
from tensorflow.keras import layers, models
import matplotlib.image as mpimg
from rsidea.preprocess import read_data, read_label, split_data


# LeNet建模
def LeNet(input_shape, output_shape):
    # 设置模型各层（卷积-池化-卷积-池化-全连接）
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_shape, padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu',
                            kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(output_shape, activation='softmax'))
    return model


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
