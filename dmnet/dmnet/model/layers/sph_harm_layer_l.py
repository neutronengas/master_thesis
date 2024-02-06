import tensorflow as tf
import numpy as np
import math
from scipy.special import binom
from tensorflow.keras import layers
from ..layers.sph_harm_layer_ml import SphHarmLayerml

class SphHarmLayerl(layers.Layer):
    def __init__(self, L, name='sphharml', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.L = L
        self.m_layers = [SphHarmLayerml(m, L) for m in range(-L, L + 1)]

    def call(self, inputs):
        # r: (None, 3)
        r = inputs
        res = tf.stack([m_layer(r) for m_layer in self.m_layers], axis=1)[:, None, :]
        return res