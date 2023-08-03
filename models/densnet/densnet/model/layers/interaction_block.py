import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .custom_dense_layer import CustomDenseLayer

class InteractionBlock(layers.Layer):
    def __init__(self, ao_vals, activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ao_vals = ao_vals

        
        self.dense_1 = CustomDenseLayer(self.ao_vals, activation)
        self.dense_2 = CustomDenseLayer(self.ao_vals, activation)
        self.dense_3 = CustomDenseLayer(self.ao_vals, activation)


    def call(self, inputs):
        out, coords_neighbors_idx, n_batch, n_grid, n_ao = inputs
        #out_transformed = self.dense_1(out)
        out_transformed = out
        n_batch, n_grid, n_ao = out_transformed.shape
        out_dummy = np.zeros(shape=(n_batch, n_grid + 1, n_ao))
        out_dummy[:, :n_grid, :] = out_transformed
        messages = tf.gather(out_dummy, coords_neighbors_idx)
        messages = tf.reduce_sum(messages, axis=-2)
        messages = self.dense_2(messages)
        out = out_transformed + messages
        out = self.dense_3(out)
        return out