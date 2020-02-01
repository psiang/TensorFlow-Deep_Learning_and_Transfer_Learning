import matplotlib.pyplot as plt
import numpy as np


# 画预测图像
def draw_predict_demo(model, x, label_name):
    plt.figure(figsize=(10, 5))
    # 显示测试图像于左侧
    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x, cmap=plt.cm.binary)
    # 获取预测
    x = np.expand_dims(x, axis=0)
    y = model.predict(x).flatten()
    # 预测数据处理
    label_num = len(label_name)
    if label_num < 5:  # 最多只显示5个预测结果
        max_pos = y.argsort()[-label_num:]
    else:
        max_pos = y.argsort()[-5:]  # 获取预测大小前五的位置
        label_num = 5
    # 显示预测指标
    ax = plt.subplot(1, 2, 1)
    plt_pos = np.arange(label_num)
    plt.barh(plt_pos, y[max_pos], align='center')  # 使用numpy高级索引可以直接抽取对应位置array的值array
    plt.yticks(plt_pos, label_name[max_pos])
    plt.xticks([])
    plt.title('Prediction')
    plt.grid(False)
    # 为每个条形图添加数值标签
    for a, b in zip(y[max_pos], plt_pos):
        plt.text(0.15, b, '%.2f%%' % (a * 100), ha='center', va='center', fontsize=11)
    # 删除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()


# 画accuracy折线图
def draw_accuracy(history):
    # 图像生成
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.show()


# 画loss折线图
def draw_loss(history):
    # 图像生成
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()