from rsidea.preprocess import read_data, read_label, split_data
from rsidea.model import *
import matplotlib.image as mpimg
from tensorflow.keras import models

save = False

"""AlexNet demo"""
# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(x, y)
# 获取未训练模型
model = AlexNet(input_shape=x_train[0].shape, output_shape=12)
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 获取数据
model.load_weights(".\\model_data\\weight\\alexnet_SIRI_WHU.h5")
history = load_history(".\\model_data\\history\\alexnet_SIRI_WHU.json")
# # 填入数据进行训练并保存模型
# history = model.fit(x_train, y_train, epochs=1,
#                     validation_data=(x_test, y_test))
# history = history.history
# # 模型保存
# if save:
#     model.save(".\\model_data\\rsidea\\alexnet_SIRI_WHU.h5")
#     model.save_weights(".\\model_data\\weight\\alexnet_SIRI_WHU.h5")
#     save_history(".\\model_data\\history\\alexnet_SIRI_WHU.json", history)
#     print("Save done!")
# 模型评测
model.evaluate(x_test, y_test, verbose=2)
# 画图
draw_accuracy(history)
draw_loss(history)
# 单张预测
names = read_label.read_SIRI_WHU()
img = mpimg.imread(".\\data\\demo.jpg") / 255.0  # 读取图像数据
draw_predict_demo(model, img, names)
