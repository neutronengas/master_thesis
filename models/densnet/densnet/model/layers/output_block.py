import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from .custom_dense_layer import CustomDenseLayer
from .custom_inner_prodcut_layer import CustomInnerProductLayer

class OutputBlock(layers.Layer):
    def __init__(self, ao_vals, activation=None, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = CustomDenseLayer(ao_vals, activation)
        self.inner_prod = CustomInnerProductLayer(ao_vals, activation)

    def call(self, inputs):
        out, _ = inputs
        out = self.dense(out)
        out = self.inner_prod(out)
        return out