from preprocess import read_data
from preprocess import read_label
from preprocess import split_data
from model import lenet
import numpy as np
import matplotlib.image as mpimg

# # 读取数据
# x, y = read_data.read_SIRI_WHU()
# # 分割数据
# x_train, y_train, x_test, y_test = split_data.split(x, y)
# # 模型训练
# lenet.train(x_train, y_train, x_test, y_test)
# # 模型评测
# lenet.evaluate(x_test, y_test)
# # 画图
# lenet.draw_loss()
# # 单张预测
names = read_label.read_SIRI_WHU()
img = mpimg.imread(".\\demo.jpg") / 255.0  # 读取图像数据
lenet.draw_predict_demo(img, names)
