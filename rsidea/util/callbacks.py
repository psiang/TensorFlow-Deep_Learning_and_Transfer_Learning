from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K


def lr_dann(model, epochs):
    # 按照论文公式设置学习率
    def scheduler(epoch):
        u0 = 0.01
        a = 10.0
        b = 0.75
        p = epoch / (epochs * 1.0)
        lr = u0 / ((1 + a * p) ** b)
        K.set_value(model.optimizer.lr, lr * 0.1)
        return K.get_value(model.optimizer.lr)
    # 形成处理学习率的Callback
    reduce_lr = LearningRateScheduler(scheduler)
    return reduce_lr
