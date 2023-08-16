import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .custom_dense_layer import CustomDenseLayer

class InteractionBlock(layers.Layer):
    def __init__(self, num_grid_points, num_features, activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_grid_points = num_grid_points
        self.num_features = num_features
        self.activation = activation

    def build(self, shape):
        self.message_weight = self.add_weight(name="message_weight", shape=(self.num_features, self.num_features))
        self.message_bias = self.add_weight(name="message_bias", shape=(self.num_features, self.num_features))
        self.update_weight =self.add_weight(name="message_weight", shape=(self.num_features, self.num_features))
        self.update_bias = self.add_weight(name="message_bias", shape=(self.num_features, self.num_features))

    def call(self, inputs):
        out, adj_matrix = inputs
        messages = tf.einsum("nij,kj->nik", out, self.message_weight)
        messages += self.message_bias
        if self.activation is not None:
            messages = self.activation(messages)
        messages = tf.einsum("njk,nij->nik", messages, adj_matrix)
        out = out + messages
        out =  tf.einsum("nij,kj->nik", out, self.update_weight)
        out += self.update_bias
        if self.activation is not None:
            out = self.activation(out)
        return out