import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .activations import swish

class DumbNet(tf.keras.Model):
    def __init__(self, activation=swish, output_init='zeros', name='dumbconvnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.flatten = Flatten()
        self.dense_layers = []
        for _ in range(1):
            self.dense_layers.append(Dense(25, activation=activation))

    def call(self, inputs):
        densities = inputs['densities']
        out = self.flatten(densities)
        for layer in self.dense_layers:
            out = layer(out)
        out = tf.reshape(out, (-1, 5, 5, 1))
        return out