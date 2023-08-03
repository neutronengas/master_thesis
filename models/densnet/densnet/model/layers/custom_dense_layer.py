import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class CustomDenseLayer(layers.Layer):
    def __init__(self, ao_vals, activation=None,
                 name='matact', **kwargs):
        super().__init__(name=name, **kwargs)
        self.activation = activation
        self.ao_vals = ao_vals
        
    def build(self, shape):
        self.weight = self.add_weight("weight", shape=(self.ao_vals, self.ao_vals))
        self.bias = self.add_weight("bias", shape=(self.ao_vals,))

    def call(self, input):
        out = tf.einsum("nga,ma->ngm", input, self.weight)
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out