import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.embedding_block import EmbeddingBlock
from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .activations import swish

class DMNet(tf.keras.Model):
    def __init__(self, num_interaction_blocks, emb_size, num_grid_points, s_type_per_atom, p_type_per_atom, activation=swish, output_init='zeros', name='densnet', **kwargs):
        super().__init__(name=name, **kwargs)
        self.embedding_block = EmbeddingBlock(emb_size)
        self.int_layers = []
        for _ in range(num_interaction_blocks):
            int_layer = InteractionBlock(emb_size, num_transforms=1, activation=activation)
            self.int_layers.append(int_layer)
        self.output_layer = OutputBlock(num_grid_points, s_type_per_atom, p_type_per_atom, emb_size, activation)

    def call(self, inputs):
        Z = inputs['Z']
        N = inputs['N']
        R = inputs['R']
        edge_id_i = inputs['edge_id_i']
        edge_id_j = inputs['edge_id_j']
        coords = inputs['coords']

        out = self.embedding_block(Z)
        for layer in self.int_layers:
            out = layer((out, edge_id_i, edge_id_j))
        out = self.output_layer((out, Z, R, coords, N))
        return out