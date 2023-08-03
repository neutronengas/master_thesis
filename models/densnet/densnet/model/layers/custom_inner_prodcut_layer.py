import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

class CustomInnerProductLayer(layers.Layer):
    def __init__(self, ao_vals, activation=None,
                 name='matact', **kwargs):
        super().__init__(name=name, **kwargs)
        self.activation = activation
        self.ao_vals = ao_vals
        
    def build(self):
        self.weight = self.add_weight("weight", shape=(self.ao_vals,))
        self.bias = self.add_weight("bias", shape=(self.ao_vals,))

    def call(self, input):
        out = tf.einsum("na,ma->nm", input, self.weight)
        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out