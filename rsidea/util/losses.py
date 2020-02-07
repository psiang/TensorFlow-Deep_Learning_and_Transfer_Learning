import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy


# ddc损失
def loss_ddc(mmd):
    def loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred) + 0.25 * mmd * mmd
    return loss


# MMD损失计算
def loss_mmd(x, y):
    xx = __gaussian_kernel(x, x)
    xy = __gaussian_kernel(x, y)
    yy = __gaussian_kernel(y, y)
    loss = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
    return loss


# 高斯核函数的计算
def __gaussian_kernel(x1, x2, beta=1.0):
    # 中间扩展1维
    r = tf.expand_dims(x1, 1)
    # 得到一个n*m的矩阵，为高斯函数的幂
    power = -beta * K.sum(K.square(r - x2), axis=-1)
    return K.exp(power)
