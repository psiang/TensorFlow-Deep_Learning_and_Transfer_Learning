import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# 读取SIRI_WHU的数据
def read_SIRI_WHU():
    data_dir = './data/Google dataset of SIRI-WHU_earth_im_tiff/12class_tif'
    datas = []
    labels = []
    # 枚举种类
    for fname_label in os.listdir(data_dir):
        # 种类文件位置
        cate_dir = os.path.join(data_dir, fname_label)
        # 枚举种类的图像
        for fname_pic in os.listdir(cate_dir):
            # 图像文件位置
            pic_dir = os.path.join(cate_dir, fname_pic)
            img = mpimg.imread(pic_dir)  # 读取图像数据
            # print(img.shape)
            # plt.imshow(img)
            # plt.show()
            data = img / 255.0  # 数据归一化
            label = fname_pic  # 获取标签
            datas.append(data)
            labels.append(label)
    # 转换成numpy的格式
    datas = np.array(datas)
    labels = np.array(labels)
    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels
