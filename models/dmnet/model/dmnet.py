import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.embedding_block import EmbeddingBlock
from .layers.interaction_block import InteractionBlock
from .layers.output_block import OutputBlock
from .activations import swish

class DMNet(tf.keras.Model):
    def __init__(self, emb_size, num_interaction_blocks, activation=swish, output_init='zeros', name='dmnet', **kwargs):
        super().__init__(name=name, **kwargs)
        # hard-coded for cc-pvdz basis
        self.no_orbitals_per_atom = 14
        self.embedding_block = EmbeddingBlock(emb_size, no_orbitals_per_atom=self.no_orbitals_per_atom)
        self.int_layers = []
        for _ in range(num_interaction_blocks):
            int_layer = InteractionBlock(emb_size, no_orbitals_per_atom=self.no_orbitals_per_atom, num_transforms=1, activation=activation)
            self.int_layers.append(int_layer)
        self.output_layer = OutputBlock(emb_size)


    def call(self, inputs):
        Z = inputs['Z']
        N = inputs['N']
        N_rdm = inputs['N_rdm']
        R = inputs['R']
        edge_id_i = inputs['edge_id_i']
        edge_id_j = inputs['edge_id_j']
        atom_pair_indices = inputs["atom_pair_indices"]
        atom_pair_mol_id = inputs["atom_pair_mol_id"]

        rdm = inputs['rdm']
        # coords = inputs['coords']

        out = self.embedding_block(Z)
        for layer in self.int_layers:
            out = layer((out, R, edge_id_i, edge_id_j))
        out = self.output_layer((out, Z, R, N, atom_pair_indices, atom_pair_mol_id, rdm, N_rdm))
        return out