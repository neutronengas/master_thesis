import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class InitLayer(layers.Layer):
    def __init__(self, num_grid_points, num_features, activation=None, name='init_layer', **kwargs):
        super().__init__(name=name, **kwargs) 
        self.num_grid_points = num_grid_points
        self.activation = activation
        self.num_features = num_features

    def build(self, shape):
        self.weight = self.add_weight(name="init_weight", shape=(self.num_grid_points, self.num_features))

    def call(self, inputs):
        densities = inputs
        densities_row_reshaped = densities[:, np.newaxis, :]
        densities_column_reshaped = densities[:, :, np.newaxis]
        init = densities_column_reshaped * densities_row_reshaped
        init = tf.einsum("nij,jk->nik", init, self.weight)
        if self.activation is not None:
            init = self.activation(init)
        return init