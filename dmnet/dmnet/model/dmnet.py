import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape

from .layers.embedding_layer import EmbeddingLayer
from .layers.cgc_layer import CgcLayer
#from .layers.interaction_block import InteractionBlock
from .layers.interaction_layer import InteractionLayer
#from .layers.output_block import OutputBlock
from .layers.output_layer import OutputLayer
from .activations import swish

class DMNet(tf.keras.Model):
    def __init__(self, F, L, K, r_cut, num_interaction_blocks, atoms, activation=swish, output_init='zeros', name='dmnet', **kwargs):
        super().__init__(name=name, **kwargs)
        # hard-coded for cc-pvdz basis
        self.L = L
        self.no_orbitals_per_atom = 14
        self.cgc_layer = CgcLayer(L)
        cgc = self.cgc_layer([])
        self.embedding_block = EmbeddingLayer(F, K, r_cut, cgc)
        self.int_layers = []
        for _ in range(num_interaction_blocks):
            int_layer = InteractionLayer(F=F, L=L, K=K, r_cut=r_cut, cgc=cgc)
            self.int_layers.append(int_layer)
        self.output_layer = OutputLayer(L=L, F=F, K=K, r_cut=r_cut, cgc=cgc, atoms=atoms)


    def call(self, inputs):
        Z = inputs['Z']
        atom_idx = inputs['atom_idx']
        pair_idx = inputs['pair_idx']
        N = inputs['N']
        N_rdm = inputs['N_rdm']
        R = inputs['R']
        atom_pair_indices = inputs["atom_pair_indices"]
        atom_pair_mol_id = inputs["atom_pair_mol_id"]
        Y = [inputs[f"Y_{l}"] for l in range(self.L + 1)]
        rdm = inputs['rdm']
        # coords = inputs['coords']

        atoms, pairs = self.embedding_block((Z, R, atom_pair_indices))
        for layer in self.int_layers:
            atoms, pairs = layer((atoms, pairs, atom_pair_indices, R, Y))
        rdm = self.output_layer((atoms, pairs, Z, R, N, atom_pair_indices, atom_pair_mol_id, atom_idx, pair_idx, rdm, N_rdm))
        return rdm