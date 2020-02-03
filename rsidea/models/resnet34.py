from tensorflow.keras.layers import *
from tensorflow.keras.models import *


# ResNet-34
def ResNet34(input_shape, output_shape):
    inpt = Input(shape=input_shape)
    # 卷积池化
    x = ZeroPadding2D((3, 3))(inpt)
    x = __Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Block(64)
    x = __Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # Block(128)
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # Block(256)
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # Block(512)
    x = __Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = __Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = __Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=inpt, outputs=x)
    return model


# Block
def __Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), is_projection=False):
    x = __Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = __Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if is_projection:
        shortcut = __Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


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
