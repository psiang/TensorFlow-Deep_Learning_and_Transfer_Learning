import os
import numpy as np


# 读取SIRI_WHU的标签
def read_SIRI_WHU():
    data_dir = '.\\data\\Google dataset of SIRI-WHU_earth_im_tiff\\12class_tif'
    return np.array(os.listdir(data_dir))