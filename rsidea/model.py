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
    model = vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape)
    tensor = Flatten(name='flatten')(model.output) # 扁平化
    tensor = Dense(4096, activation='relu', name='fc1')(tensor)
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(4096, activation='relu', name='fc2')(tensor)
    tensor = Dropout(0.5)(tensor)
    tensor = Dense(output_shape, activation='softmax')(tensor)
    model_vgg = Model(inputs=model.input, outputs=tensor, name='vgg16')
    return model_vgg
