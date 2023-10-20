import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

class InteractionBlock(layers.Layer):
    def __init__(self, emb_size, num_transforms, activation=None, name='interaction', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.num_transforms = num_transforms
        self.activation = activation
        self.dense_1 = Dense(emb_size, activation='relu')
        self.dense_2 = Dense(emb_size, activation='relu')
        self.dense_3 = Dense(emb_size, activation='relu')

    def build(self, shape):
        self.transform_mat_weights = self.add_weight(name="mat_weight", shape=(self.emb_size, self.emb_size), dtype=tf.float32, trainable=True)
        self.transform_biases = self.add_weight(name="bias", shape=(self.emb_size,), dtype=tf.float32, trainable=True)


    def call(self, inputs):
        # out: (None, self.emb_size); edge_id_i: (None,), edge_id_j: (None,)
        out, edge_id_i, edge_id_j = inputs
        msg = tf.gather(out, edge_id_i)
        #msg = self.dense_1(msg)
        msg = tf.einsum("ni,ik->nk", msg, self.transform_mat_weights)
        msg = msg + self.transform_biases
        msg = self.activation(msg)
        #print(edge_id_i.shape)
        msg = tf.math.unsorted_segment_sum(msg, edge_id_j, num_segments=len(out))
        msg = self.dense_2(msg)
        out = out + msg
        out = self.dense_3(out)
        return out