import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.hartree_fock_layer import HartreeFockLayer
from .layers.init_layer import InitLayer
from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .activations import swish

class DensNet(tf.keras.Model):
    def __init__(self, ao_vals, num_interaction_blocks, num_grid_points, activation=swish, output_init='zeros', name='densnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_grid_points = num_grid_points
        self.hf = HartreeFockLayer()
        self.init_layer = InitLayer(num_grid_points)
        self.int_layers = []
        for _ in range(num_interaction_blocks):
            int_layer = InteractionBlock(num_grid_points, activation)
            self.int_layers.append(int_layer)
        #self.output_layer = OutputBlock(ao_vals, num_grid_points, activation)

    def call(self, inputs):
        R = inputs['R']
        coords = inputs['coords']
        Z = tf.ones(shape=(2,), dtype=tf.int32) # hardcoded for hydrogen
        coords_neighbors_idx = inputs['neighbour_coords_idx']
        out = self.init_layer((R, coords, Z))
        out = (out, coords_neighbors_idx)
        for layer in self.int_layers:
            out = layer(out)
        #out = self.output_layer(out)
        out, _ = out
        return out