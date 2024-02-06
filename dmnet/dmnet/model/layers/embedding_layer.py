import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from ..layers.pairmix_layer import PairmixLayer


class EmbeddingLayer(layers.Layer):
    def __init__(self, F, K, r_cut, cgc, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = F
        self.weight_init = tf.keras.initializers.GlorotUniform()

        # for now 14 different atoms are assumed
        emb_init = tf.initializers.GlorotNormal()
        
        self.leq0_embeddings = self.add_weight(name="l0_embeddings", shape=(14, self.F, 1), dtype=tf.float32, initializer=emb_init, trainable=True)
        self.leq1_embeddings = self.add_weight(name="l1_embeddings", shape=(14, self.F, 3), dtype=tf.float32, initializer=emb_init, trainable=True)
        self.leq2_embeddings = self.add_weight(name="l2_embeddings", shape=(14, self.F, 5), dtype=tf.float32, initializer=emb_init, trainable=True)
        self.leq3_embeddings = tf.zeros(shape=(14, self.F, 7))
        self.leq4_embeddings = tf.zeros(shape=(14, self.F, 9))
        self.leq5_embeddings = tf.zeros(shape=(14, self.F, 11))

        self.pair_mix = PairmixLayer(5, 5, 5, F, K, r_cut, cgc)   

    def call(self, inputs):
        Z, R, atom_pair_indices = inputs
        R_ij = tf.norm(tf.gather(R, atom_pair_indices[:, 0]) - tf.gather(R, atom_pair_indices[:, 1]), axis=-1)
        # out: (None, self.no_orbitals_per_atom, self.embeddings)
        atoms = [
            tf.gather(self.leq0_embeddings, Z),
            tf.gather(self.leq1_embeddings, Z),
            tf.gather(self.leq2_embeddings, Z),
            tf.gather(self.leq3_embeddings, Z),
            tf.gather(self.leq4_embeddings, Z),
            tf.gather(self.leq5_embeddings, Z)
        ]
        x = [
            tf.gather(atoms[0], atom_pair_indices[:, 0]),
            tf.gather(atoms[1], atom_pair_indices[:, 0]),
            tf.gather(atoms[2], atom_pair_indices[:, 0]),
            tf.gather(atoms[3], atom_pair_indices[:, 0]),
            tf.gather(atoms[4], atom_pair_indices[:, 0]),
            tf.gather(atoms[5], atom_pair_indices[:, 0]),
        ]
        y = [
            tf.gather(atoms[0], atom_pair_indices[:, 1]),
            tf.gather(atoms[1], atom_pair_indices[:, 1]),
            tf.gather(atoms[2], atom_pair_indices[:, 1]),
            tf.gather(atoms[3], atom_pair_indices[:, 1]),
            tf.gather(atoms[4], atom_pair_indices[:, 1]),
            tf.gather(atoms[5], atom_pair_indices[:, 1]),
        ]
        pairs = self.pair_mix((x, y, R_ij))
        return atoms, pairs