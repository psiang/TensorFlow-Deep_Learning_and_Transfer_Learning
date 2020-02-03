from tensorflow.keras.layers import *
from tensorflow.keras.models import *


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
