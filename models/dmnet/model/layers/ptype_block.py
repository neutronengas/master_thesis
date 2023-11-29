import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


class PTypeBlock(layers.Layer):
    def __init__(self, p_type_exp_per_atom, name='ptypeblock', **kwargs):
        super().__init__(name=name, **kwargs)
        self.p_type_exp_per_atom = p_type_exp_per_atom

    def call(self, inputs):
        p_type_cs, p_type_alphas, v_coeffs = inputs
        # out: (None, self.embeddings)
        out = tf.gather(self.embeddings, Z)
        return out