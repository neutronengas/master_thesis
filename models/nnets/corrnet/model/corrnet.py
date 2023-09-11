import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.init_layer import InitLayer
from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .activations import swish

class CorrNet(tf.keras.Model):
    def __init__(self, ao_vals, num_interaction_blocks, num_featuers, num_grid_points, activation=swish, output_init='zeros', name='densnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_grid_points = num_grid_points
        self.num_features= num_featuers
        self.init_layer = InitLayer(num_grid_points, num_features=self.num_features)
        self.int_layers = []
        for i in range(num_interaction_blocks):
            int_layer = InteractionBlock(num_grid_points, self.num_features, activation, name=f"interaction_{str(i)}")
            self.int_layers.append(int_layer)
        self.output_layer = OutputBlock(num_grid_points, self.num_features, activation)

    def call(self, inputs):
        adj_matrix = inputs['adj_matrix']
        densities = inputs['densities']
        out = self.init_layer((densities))
        for layer in self.int_layers:
            out = layer((out, adj_matrix))
        out = self.output_layer(out)
        return out