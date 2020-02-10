import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model

from rsidea.util.layers import GradientReversal


# DANN建模
def DANN(input_shape, output_shape, batch_size=32):
    # 输入
    inputs = Input(shape=input_shape, name="source_input")
    # 通过特征提取器
    feature_extractor = __feature_extractor(input_shape=inputs[0].shape)
    feature_output = feature_extractor(inputs)
    # 通过判别器
    domain_classifier = __domain_classifier(input_shape=feature_output[0].shape)
    domain_classifier_output = domain_classifier(feature_output)
    # 通过分类器
    source_feature = Lambda(lambda x: K.switch(K.learning_phase(),
                                               K.concatenate([x[:int(batch_size // 2)], x[:int(batch_size // 2)]],
                                                             axis=0),
                                               x),
                            output_shape=lambda x: x)(feature_output)
    label_predictor = __label_predictor(input_shape=feature_output[0].shape, output_shape=output_shape)
    label_predictor_output = label_predictor(source_feature)
    # 实例化模型
    model = Model(inputs=inputs, outputs=[label_predictor_output, domain_classifier_output])
    return model


# 特征提取器（使用LeNet）
def __feature_extractor(input_shape):
    inputs = Input(shape=input_shape)
    tensor = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(inputs)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)

    tensor = Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(tensor)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)

    tensor = Dropout(0.5)(tensor)
    tensor = Flatten()(tensor)

    feature_output = Dense(100, activation="relu")(tensor)
    # 实例化提取器
    model = Model(inputs=inputs, outputs=feature_output, name='feature_extractor')
    return model


# 判别器
def __domain_classifier(input_shape):
    inputs = Input(shape=input_shape)
    # 插入GRL
    grl_layer = GradientReversal()
    out = grl_layer(inputs)
    out = Dense(128, activation="relu")(out)
    out = Dropout(0.5)(out)
    classifier_output = Dense(2, activation="softmax", name="discriminator_output")(out)
    # 实例化判别器
    model = Model(inputs=inputs, outputs=classifier_output, name='domain_classifier')
    return model


# 预测器
def __label_predictor(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    out = Dense(128, activation="relu")(inputs)
    out = Dropout(0.5)(out)
    predictor_output = Dense(output_shape, activation="softmax", name="classifier_output")(out)
    # 实例化预测器
    model = Model(inputs=inputs, outputs=predictor_output, name='label_predictor')
    return model


# 数据处理
def preprocess_data(x_source, y_source, x_target, batch_size=32):
    index = 0
    half_batch = int(batch_size // 2)
    images = []
    domains = []
    truths = []
    while x_source.shape[0] > index + half_batch:
        batch_images = np.concatenate((x_source[index: index + half_batch],
                                       x_target[index: index + half_batch]), axis=0)
        batch_domains = np.concatenate((np.array([0] * half_batch),
                                        np.array([1] * half_batch)), axis=0)
        batch_truths = np.concatenate((y_source[index: index + half_batch],
                                       y_source[index: index + half_batch]), axis=0)
        if index == 0:
            images = batch_images
            domains = batch_domains
            truths = batch_truths
        else:
            images = np.concatenate((images, batch_images), axis=0)
            domains = np.concatenate((domains, batch_domains), axis=0)
            truths = np.concatenate((truths, batch_truths), axis=0)
        index += half_batch
    return images, truths, domains

