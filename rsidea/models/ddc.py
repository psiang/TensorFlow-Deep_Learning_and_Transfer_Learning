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
    model = Model(inputs=[inputs_1, inputs_2], outputs=tensor, name='ddc')
    return model, mmd


