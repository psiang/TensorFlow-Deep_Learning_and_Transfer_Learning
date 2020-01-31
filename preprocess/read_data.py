import os
import matplotlib.image as mpimg
import numpy as np


# 读取SIRI_WHU的数据
def read_SIRI_WHU():
    data_dir = '.\\data\\Google dataset of SIRI-WHU_earth_im_tiff\\12class_tif'
    datas = []
    labels = []
    label_count = 0  # 产生标签编号
    try:
        # 枚举种类
        for fname_label in os.listdir(data_dir):
            cate_dir = os.path.join(data_dir, fname_label)  # 种类文件位置
            # 枚举种类的图像
            for fname_pic in os.listdir(cate_dir):
                pic_dir = os.path.join(cate_dir, fname_pic)  # 图像文件位置
                img = mpimg.imread(pic_dir)  # 读取图像数据
                # print(img.shape)
                # plt.imshow(img)
                # plt.show()
                data = img / 255.0  # 数据归一化
                label = label_count  # 获取标签（为了符合神经网络标签必须是数字）
                datas.append(data)
                labels.append(label)
                if data.shape != datas[0].shape:  # 像素大小不一致
                    raise Exception("Image pixel sizes are inconsistent!", pic_dir)
            label_count += 1
    except Exception as err:
        # 像素大小不一致会使得神经网络不方便进行学习
        print(err.args[0], "In:", err.args[1])
    # 转换成numpy的格式
    datas = np.array(datas)
    labels = np.array(labels)
    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels
