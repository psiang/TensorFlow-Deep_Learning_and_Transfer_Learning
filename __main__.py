from preprocess import read_data
from preprocess import split_data
from model import CNN
import numpy as np


# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(x, y)
print(x_train[0].shape)
# 模型运算
CNN.run(x_train, y_train, x_test, y_test)

# a = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
# a = a.reshape(-1,2,2)
# print(a)
# print(a.shape)