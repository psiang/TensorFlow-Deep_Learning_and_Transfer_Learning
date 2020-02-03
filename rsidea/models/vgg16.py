from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import vgg16


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
