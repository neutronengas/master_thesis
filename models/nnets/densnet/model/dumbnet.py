import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .activations import swish

class DumbNet(tf.keras.Model):
    def __init__(self, ao_vals, num_interaction_blocks, num_grid_points, activation=swish, output_init='zeros', name='densnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv2d_layers = []
        for _ in range(3):
            self.conv2d_layers.append(Conv2D(1, 3, activation='relu', padding='same', input_shape=(None, 90, 30, 1)))


    def call(self, inputs):
        R = inputs['R']
        coords = inputs['coords']
        coords_neighbors_idx = inputs['neighbour_coords_idx']
        out = tf.ones(shape=(len(R), 90, 30, 1))
        for layer in self.conv2d_layers:
            out = layer(out)
        out = tf.reshape(out, (len(R), 900, 3))
        out = tf.einsum("nij->ni", out)
        return out