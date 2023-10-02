import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


class EmbeddingBlock(layers.Layer):
    def __init__(self, emb_size, activation=None,
                 name='embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.emb_size = emb_size
        self.weight_init = tf.keras.initializers.GlorotUniform()

        # for now 14 different atoms are assumed
        emb_init = tf.initializers.RandomUniform(minval=-np.sqrt(3), maxval=np.sqrt(3))
        self.embeddings = self.add_weight(name="embeddings", shape=(14, self.emb_size),
                                          dtype=tf.float32, initializer=emb_init, trainable=True)
        
        self.dense_rdm = layers.Dense(self.emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_init)
        self.dense = layers.Dense(self.emb_size, activation=activation, use_bias=True, kernel_initializer=self.weight_init)


    def call(self, inputs):
        Z = inputs
        # out: (None, self.embeddings)
        out = tf.gather(self.embeddings, Z)
        return out