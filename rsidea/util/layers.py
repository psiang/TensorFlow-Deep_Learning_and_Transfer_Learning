import tensorflow as tf
from tensorflow.keras.engine import Layer
import tensorflow.keras.backend as K


class GradientReversal(Layer):
    def __init__(self, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.num_calls = 0

    def call(self, x, mask=None):
        grad_name = "GradientReversal%d" % self.num_calls

        @tf.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * 1.0]

        g = K.get_session().graph
        with g.gradient_override_map({'Identity': grad_name}):
            y = tf.identity(x)

        self.num_calls += 1

        return y

