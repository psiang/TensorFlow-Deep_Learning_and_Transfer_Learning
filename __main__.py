import matplotlib.image as mpimg

from rsidea.models import *
from rsidea.preprocess import read_data, read_label, split_data
from rsidea.util.callbacks import lr_dann
from rsidea.util.draw import *
from rsidea.util.history import *
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
from rsidea.util.losses import loss_dann

SAVE = False
BATCH_SIZE = 32
EPOCH = 10

'''dann demo'''
# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_source, y_source, x_target, y_target = split_data.split(x, y, rate=0.5)
images, truths, domains = dann.preprocess_data(x_source[:300], y_source[:300], x_target[:300], batch_size=BATCH_SIZE)
print(images.shape)
# 获取原训练模型
model = dann.DANN(input_shape=x_source[0].shape, output_shape=12, batch_size=BATCH_SIZE)
model.summary()
# 配置模型
model.compile(optimizer='adam',
              loss={'label_predictor': 'sparse_categorical_crossentropy',
                    'domain_classifier': 'sparse_categorical_crossentropy'},
              loss_weights={'label_predictor': 1.0,
                            'domain_classifier': 0.31},
              metrics=['accuracy'],
              experimental_run_tf_function=False)
# # 填入数据进行训练
# history = model.fit(images, [truths, domains], epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=[lr_dann(model, EPOCH)])
# history = history.history
# # 模型保存
# if SAVE:
#     model.save(".\\model_data\\model\\ddc_SIRI_WHU.h5")
#     model.save_weights(".\\model_data\\weight\\ddc_SIRI_WHU.h5")
#     save_history(history, ".\\model_data\\history\\ddc_SIRI_WHU.json")
#     print("Saved!")
# # 模型评测
# model.evaluate(x_test, y_test, verbose=2)
# # 画折线图
# draw_accuracy(history)
# draw_loss(history)
# # 画单张预测展示图
# label_name = read_label.read_SIRI_WHU()
# x = mpimg.imread(".\\data\\demo.jpg") / 255.0
# draw_predict_demo(model, x, label_name)
