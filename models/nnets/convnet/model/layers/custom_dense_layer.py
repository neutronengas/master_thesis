import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class CustomDenseLayer(layers.Layer):
    def __init__(self, num_grid_points, activation=None,
                 name='matact', **kwargs):
        super().__init__(name=name, **kwargs)
        self.activation = activation
        self.num_grid_points = num_grid_points
        
    def build(self, shape):
        self.weight = self.add_weight("weight", shape=(self.num_grid_points,), trainable=True)
        self.bias = self.add_weight("bias", shape=(self.num_grid_points,), trainable=True)

    def call(self, input):
        out = self.weight * input
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out