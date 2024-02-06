from tensorflow.keras import layers
import tensorflow as tf

from ..layers.tensor_product_contraction import TensorProductContractionLayer

class SelfmixLayer(layers.Layer):
    def __init__(self, cgc, Lin, Lout, Fin, name='selfmix', **kwargs):
        super().__init__(name=name, **kwargs)

        self.tens_prod_cont = TensorProductContractionLayer(cgc)

        self.Lin = Lin
        self.Lout = Lout
        self.Fin = Fin

    def build(self, shape):
        Lout = self.Lout
        Fin = self.Fin
        Lin = self.Lin    
        self.k = self.add_weight(name="k", shape=(Lout + 1, Fin), dtype=tf.float32, initializer="glorot_uniform", trainable=True)
        self.s = self.add_weight(name="s", shape=(int((Lout + 1) * Lin * (Lin + 1) / 2), Fin), dtype=tf.float32, initializer="glorot_uniform", trainable=True)


    def call(self, inputs):
        x = inputs
        res = []
        for l in range(self.Lout + 1):
            appendix = x[l] * tf.gather(self.k, [l]) if l <= self.Lin else 0
            for l1 in range(self.Lin + 1):
                for l2 in range(l1 + 1, self.Lin + 1):
                        self.tens_prod_cont.set_params(l1, l2, l)
                        appendix += tf.gather(self.s, [l1 * self.Lin + l2]) * self.tens_prod_cont([x, x])
            res.append(appendix)
        return res