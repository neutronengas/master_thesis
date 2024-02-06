from tensorflow.keras import layers
import tensorflow as tf

class LinearLayer(layers.Layer):
    def __init__(self, Fin, Fout, name='linear', **kwargs):
        super().__init__(name=name, **kwargs)
        self.Fin = Fin
        self.Fout = Fout


    def build(self, shape):
        Fout = self.Fout
        Fin = self.Fin
        initializer = tf.keras.initializers.GlorotNormal()
        self.l0_W = self.add_weight(name="l0_W", shape=(Fout, Fin), dtype=tf.float32, initializer=initializer, trainable=True)
        self.l1_W = self.add_weight(name="l1_W", shape=(Fout, Fin), dtype=tf.float32, initializer=initializer, trainable=True)
        self.l2_W = self.add_weight(name="l2_W", shape=(Fout, Fin), dtype=tf.float32, initializer=initializer, trainable=True)
        self.l3_W = self.add_weight(name="l3_W", shape=(Fout, Fin), dtype=tf.float32, initializer=initializer, trainable=True)
        self.l4_W = self.add_weight(name="l4_W", shape=(Fout, Fin), dtype=tf.float32, initializer=initializer, trainable=True)
        self.l5_W = self.add_weight(name="l5_W", shape=(Fout, Fin), dtype=tf.float32, initializer=initializer, trainable=True)
        self.b = self.add_weight(name="b", shape=(Fout, 1), initializer=initializer)
        
    def call(self, inputs):
        [l0_embeddings, l1_embeddings, l2_embeddings, l3_embeddings, l4_embeddings, l5_embeddings] = inputs
        l0_embeddings = tf.einsum("nij,ki->nkj", l0_embeddings, self.l0_W) + self.b
        l1_embeddings = tf.einsum("nij,ki->nkj", l1_embeddings, self.l1_W)
        l2_embeddings = tf.einsum("nij,ki->nkj", l2_embeddings, self.l2_W)
        l3_embeddings = tf.einsum("nij,ki->nkj", l3_embeddings, self.l3_W)
        l4_embeddings = tf.einsum("nij,ki->nkj", l4_embeddings, self.l4_W)
        l5_embeddings = tf.einsum("nij,ki->nkj", l5_embeddings, self.l5_W)
        return [l0_embeddings, l1_embeddings, l2_embeddings, l3_embeddings, l4_embeddings, l5_embeddings]