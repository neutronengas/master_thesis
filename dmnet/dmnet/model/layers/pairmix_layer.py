from tensorflow.keras import layers
import tensorflow as tf

from ..layers.bernstein_layer import BernsteinLayer
from ..layers.tensor_product_contraction import TensorProductContractionLayer


class PairmixLayer(layers.Layer):
    def __init__(self, Lx, Ly, Lout, F, K, r_cut, cgc, name='residual', **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.F = F
        self.K = K
        self.Lx = Lx
        self.Ly = Ly
        self.Lout = Lout
        self.n_matrices = (Lx + 1) * (Ly + 1) * (Lout + 1)
        self.initializer = tf.keras.initializers.GlorotNormal()

        self.bernstein_layer = BernsteinLayer(K, r_cut)
        self.tens_prod_layer = TensorProductContractionLayer(cgc)


    def build(self, shape):
        self.weight_matrices = self.add_weight(name="weights", shape=(self.n_matrices, self.F, self.K), initializer=self.initializer)
                

    def call(self, inputs):
        x, y, r = inputs
        out = []
        for l3 in range(self.Lout + 1):
            appendix = 0
            for l1 in range(self.Lx + 1):
                for l2 in range(self.Ly + 1):
                    idx = l1 * (self.Ly + 1) * (self.Lout + 1) + l2 * (self.Lout + 1)  + l3
                    weight_mtx = self.weight_matrices[idx]
                    g = self.bernstein_layer(r)
                    self.tens_prod_layer.set_params(l1, l2, l3)
                    tp_output = self.tens_prod_layer([x, y])
                    mul = tf.einsum("ij,nj->ni", weight_mtx, g)[:, :, None]
                    a = tp_output * mul
                    appendix += a

            out.append(appendix)
        return out