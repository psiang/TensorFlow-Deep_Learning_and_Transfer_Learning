import matplotlib.image as mpimg
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from rsidea.preprocess import read_data, read_label, split_data
from rsidea.util.draw import *
from rsidea.util.history import *


# 建立预训练的网络
def build_model(input_shape, output_shape):
    # 为了改输出层得到去掉全连接层的预训练InceptionV3
    model = inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    # 添加新的全连接层
    tensor = GlobalAveragePooling2D()(model.output)
    tensor = Dense(1024, activation='relu')(tensor)
    tensor = Dense(output_shape, activation='softmax')(tensor)
    model_v3 = Model(inputs=model.input, outputs=tensor, name='inception_v3')
    return model_v3


save = True

'''finetune demo'''
# 读取数据
x, y = read_data.read_SIRI_WHU()
# 分割数据
x_train, y_train, x_test, y_test = split_data.split(x, y, rate=0.8)
# 获取原训练模型
model = build_model(input_shape=x_train[0].shape, output_shape=12)
# 冻结前172层
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 填入数据进行训练
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test))
history = history.history
# 模型保存
if save:
    model.save(".\\model_data\\model\\finetune_SIRI_WHU.h5")
    model.save_weights(".\\model_data\\weight\\finetune_SIRI_WHU.h5")
    save_history(history, ".\\model_data\\history\\finetune_SIRI_WHU.json")
    print("Saved!")
# 模型评测
model.evaluate(x_test, y_test, verbose=2)
# 画折线图
draw_accuracy(history)
draw_loss(history)
# 画单张预测展示图
label_name = read_label.read_SIRI_WHU()
x = mpimg.imread(".\\data\\demo.jpg") / 255.0
draw_predict_demo(model, x, label_name)