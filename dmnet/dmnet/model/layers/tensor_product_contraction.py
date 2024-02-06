from tensorflow.keras import layers
import tensorflow as tf

class TensorProductContractionLayer(layers.Layer):
    def __init__(self, cgc, name='activation', **kwargs):
        super().__init__(name=name, **kwargs)

        self.cgc = cgc

    def set_params(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def call(self, inputs):
        [x, y] = inputs
        xl1_embedding = x[self.l1]
        yl2_embedding = y[self.l2]
        if (self.l1, self.l2, self.l3) in self.cgc.keys():
            l1l2l3_cgc = self.cgc[(self.l1, self.l2, self.l3)]
        else:
            l1l2l3_cgc = tf.zeros(shape=(2*self.l1+1, 2*self.l2+1, 2*self.l3+1))
        res = tf.einsum("mnh,ifm,ifn->ifh", l1l2l3_cgc, xl1_embedding, yl2_embedding)
        return res