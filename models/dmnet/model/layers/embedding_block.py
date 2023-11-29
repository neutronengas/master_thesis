import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, no_orbitals_per_atom, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.no_orbitals_per_atom = no_orbitals_per_atom
        self.emb_size = emb_size
        self.weight_init = tf.keras.initializers.GlorotUniform()

        # for now 14 different atoms are assumed
        emb_init = tf.initializers.GlorotNormal()
        self.embeddings = self.add_weight(name="embeddings", shape=(14, self.no_orbitals_per_atom, self.emb_size),
                                          dtype=tf.float32, initializer=emb_init, trainable=True)
        
        self.dense_rdm = layers.Dense(self.emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_init)


    def call(self, inputs):
        Z = inputs
        # out: (None, self.no_orbitals_per_atom, self.embeddings)
        out = tf.gather(self.embeddings, Z)
        return out