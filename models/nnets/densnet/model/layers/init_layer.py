import tensorflow as tf
from pyscf import scf, gto
from tensorflow.keras import layers
from ..utils.gto_proxy import exponential_tensorwise
import numpy as np

class InitLayer(layers.Layer):
    def __init__(self, num_grid_points, name='init_layer', **kwargs):
        super().__init__(name=name, **kwargs) 
        self.num_grid_points = num_grid_points

    def build(self, shape):
        self.weight = self.add_weight(name="exp_matrix", shape=(3, 3))

    def call(self, inputs):
        R_tensor, coords, Z = inputs
        # bring R_tensor and coords into same shape

        # R_tensor of shape (None, Z.sum(), 3)
        R_tensor_reshaped = tf.expand_dims(R_tensor, axis=1) # (None, 1, Z.sum(), 3) 
        R_tensor_reshaped = tf.tile(R_tensor_reshaped, [1, self.num_grid_points, 1, 1]) # (None, num_grid_points, Z.sum(), 3)

        # coords of shape (num_grid_points, 3)
        coords_reshaped = tf.expand_dims(coords, axis=1) # (num_grid_points, 1, 3)
        coords_reshaped = tf.tile(coords_reshaped, [1, tf.math.reduce_sum(Z), 1]) # (num_grid_points, Z.sum(), 3)
        points_R_diff = coords_reshaped - R_tensor_reshaped # shape (None, num_grid_points, Z.sum(), 3)

        points_R_diff = tf.einsum("ijkl,ml->ijkm", points_R_diff, self.weight)
        points_R_diff = tf.norm(points_R_diff, axis=-1)
        out = tf.exp(points_R_diff)
        out = tf.math.reduce_sum(out, axis=2)
        return out