import numpy as np
import json
import codecs


# 保存历史信息为json文件
def save_history(path, history):
    new_hist = {}
    for key in list(history.keys()):
        if type(history[key]) == np.ndarray:
            new_hist[key] = history[key].tolist()
        elif type(history[key]) == list:
            if type(history[key][0]) == np.float64 or type(history[key][0]) == np.float32:
                new_hist[key] = list(map(float, history[key]))
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)


# 加载历史信息
def load_history(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n
