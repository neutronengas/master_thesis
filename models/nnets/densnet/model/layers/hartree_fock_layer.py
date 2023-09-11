import tensorflow as tf
from pyscf import scf, gto
from tensorflow.keras import layers
import numpy as np

class HartreeFockLayer(layers.Layer):
    def __init__(self, basis='ccpvdz', name='hartree_fock_layer', **kwargs):
        super().__init__(name=name, **kwargs)
        self.basis = basis
        self.dense = layers.Dense(1)

    def hartree_fock_call(self, R, coords):
        mol = gto.M()
        mol.atom = list(zip([1, 1], R))
        mol.basis = self.basis
        mol.build()
        ao_vals = np.array(mol.eval_ao("GTOval_sph", coords))
        # shape of ao_vals: (n_coords, n_aovals)
        return ao_vals
    
    def hartree_fock_call_alternative(self, R, coords):
        # implement rough initialization with GTO value estimate
        n_coords = len(coords)
        n_aovals = 10
        return np.random.random((n_coords, n_aovals))
    
    def hartree_fock_tensor_call(self, R_tensor, coords):
        n = len(R_tensor)
        return np.array([self.hartree_fock_call_alternative(R_tensor[i], coords) for i in range(n)]).astype(np.float32)


    def call(self, inputs):
        R_tensor, coords = inputs
        out = tf.numpy_function(self.hartree_fock_tensor_call, (R_tensor, coords), tf.float32)
        return out