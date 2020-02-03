from tensorflow.keras.layers import *
from tensorflow.keras.models import *


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