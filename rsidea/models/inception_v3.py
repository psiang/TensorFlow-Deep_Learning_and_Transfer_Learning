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
