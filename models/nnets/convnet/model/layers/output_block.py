import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from .custom_dense_layer import CustomDenseLayer
from .custom_inner_prodcut_layer import CustomInnerProductLayer

class OutputBlock(layers.Layer):
    def __init__(self, num_grid_points, num_features, activation=None, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_features = num_features
        self.num_grid_points = num_grid_points
        self.activation = activation

    def build(self, shape):
        self.transform_weight = self.add_weight(name="transform_weight", shape=(self.num_features,))


    def call(self, inputs):
        out = inputs
        out_a = out[:, :, np.newaxis, :]
        out_b = out[:, np.newaxis, :, :]
        out = out_a + out_b
        out = tf.einsum("nijk,k->nij", out, self.transform_weight)
        return out