import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class CustomInnerProductLayer(layers.Layer):
    def __init__(self, ao_vals, num_grid_points, activation=None,
                 name='matact', **kwargs):
        super().__init__(name=name, **kwargs)
        self.activation = activation
        self.ao_vals = ao_vals
        self.num_grid_points = num_grid_points
        
    def build(self, shape):
        self.weight = self.add_weight("weight", shape=(self.ao_vals,), trainable=True)
        self.bias = self.add_weight("bias", shape=(self.num_grid_points,), trainable=True)

    def call(self, input):
        out = tf.einsum("nia,a->ni", input, self.weight)
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out