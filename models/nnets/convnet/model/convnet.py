import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .activations import swish

class ConvNet(tf.keras.Model):
    def __init__(self, activation=swish, output_init='zeros', name='convnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv2d_layers = []
        for _ in range(3):
            self.conv2d_layers.append(Conv2D(1, 3, activation='relu', padding='same', input_shape=(None, 5, 5, 1)))


    def call(self, inputs):
        densities = inputs['densities']
        out = densities
        for layer in self.conv2d_layers:
            out = layer(out)
        return out