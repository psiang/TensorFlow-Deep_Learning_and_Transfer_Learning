from tensorflow.keras.layers import Dense, Input, Lambda, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


from rsidea.util.layers import GradientReversal


# DANN建模
def DANN(input_shape, output_shape):
    # 输入
    inputs = Input(shape=input_shape, name="main_input")
    # 通过特征提取器
    feature_output = __feature_extractor(inputs)
    # 通过判别器
    grl_layer = GradientReversal()
    feature_output_grl = grl_layer(feature_output)
    labeled_feature_output = Lambda(lambda x: K.switch(K.learning_phase(), K.concatenate(
        [x[:int(self.batch_size // 2)], x[:int(self.batch_size // 2)]], axis=0), x), output_shape=lambda x: x[0:])(
        feature_output_grl)

    classifier_output = self.classifier(labeled_feature_output)
    discriminator_output = self.discriminator(feature_output)
    model = keras.models.Model(inputs=inp, outputs=[discriminator_output, classifier_output])
    return model


# 特征提取器（使用LeNet）
def __feature_extractor(inputs):
    tensor = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(inputs)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)

    tensor = Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(tensor)
    tensor = MaxPooling2D(pool_size=(2, 2))(tensor)

    tensor = Dropout(0.5)(tensor)
    tensor = Flatten()(tensor)

    feature_output = Dense(100, activation="relu")(tensor)
    return feature_output
