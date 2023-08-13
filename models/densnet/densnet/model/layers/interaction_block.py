import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .custom_dense_layer import CustomDenseLayer

class InteractionBlock(layers.Layer):
    def __init__(self, num_grid_points, activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_grid_points = num_grid_points
        self.dense_1 = CustomDenseLayer(self.num_grid_points, activation)
        self.dense_2 = CustomDenseLayer(self.num_grid_points, activation)
        self.dense_3 = CustomDenseLayer(self.num_grid_points, activation)


    def call(self, inputs):
        out, coords_neighbors_idx = inputs
        out = self.dense_1(out)
        # messages = tf.einsum("njk,ij->nik", out, coords_neighbors_idx)
        # messages = self.dense_2(messages)
        # out = out + messages
        # out = self.dense_3(out)
        return out, coords_neighbors_idx