import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_init = tf.keras.initializers.GlorotUniform()

        # base function embedding when working in the ccpvdz basis. We have atoms H - Ne with 5 basis functions for each of H and He and 14 for each of the other atoms, giving a total of 122 basis functions
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        self.embeddings = self.add_weight(name="embeddings", shape=(122, self.emb_size),
                                          dtype=tf.float32, initializer=emb_init, trainable=True)
        
        self.dense_rdm = layers.Dense(self.emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_init)


    def call(self, inputs):
        bf_idx = inputs
        h = tf.gather(self.embeddings, bf_idx)
        return h