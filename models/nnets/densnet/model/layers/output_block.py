import numpy as np
import tensorflow as tf
import math

from tensorflow.keras import layers

class OutputBlock(layers.Layer):
    def __init__(self, num_grid_points, s_type_exp_per_atom, p_type_exp_per_atom, emb_size, activation=None, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_grid_points = num_grid_points
        self.s_type_exp_per_atom = s_type_exp_per_atom
        self.p_type_exp_per_atom = p_type_exp_per_atom
        # number of parameters needed: for each s-type orbital, there is one alpha and one c needed;
        #                              for each p-type orbital, there is one alpha, one c and 3 v-vector elements needed
        self.dense = tf.keras.layers.Dense(2 * s_type_exp_per_atom + 5 * p_type_exp_per_atom, activation=None)
        self.emb_size = emb_size     
        self.reshape = layers.Reshape((self.p_type_exp_per_atom, 3), input_shape=(self.p_type_exp_per_atom * 3,))   

    def call(self, inputs):
        # out: (None, self.emb_size); Z: (None,); R: (None, 3); coords: (None, self.num_grid_points, 3), N: (None,)
        out, Z, R, coords, N = inputs
        a = [r for r in Z.unstack()]
        print(a)
        n_mol = tf.shape(N)[0]
        mol_ids = tf.repeat(tf.range(n_mol), N)
        N_electrons = tf.math.unsorted_segment_sum(Z, mol_ids, n_mol)[:, None]
        # recreate grid points for each atom of the molecule separately
        coords = tf.repeat(coords, N, axis=0)
        # recreate atomic positions for each point on the grid
        coeffs = self.dense(out)
        #coords = tf.transpose(coords, (0, 2, 1))
        s_type_alphas = coeffs[:, :self.s_type_exp_per_atom] ** 2
        index = self.s_type_exp_per_atom
        s_type_cs = coeffs[:, index:index+self.s_type_exp_per_atom]
        index += self.s_type_exp_per_atom
        p_type_alphas = coeffs[:, index:index+self.p_type_exp_per_atom] ** 2
        index += self.p_type_exp_per_atom
        p_type_cs = coeffs[:, index:index+self.p_type_exp_per_atom]
        index += self.p_type_exp_per_atom
        v_coeffs = coeffs[:, index:]
        # calculate the integrals of the orbitals for later normalization
        s_type_integrals = s_type_cs * tf.math.sqrt(math.pi / s_type_alphas)
        p_type_integrals = 0.5 * p_type_cs * tf.math.sqrt(math.pi ** 3 / p_type_alphas ** 5)
        # create the contribution from the s_type orbitals
        # calculate the norm of the s-type orbitals 
        #s_type_out = s_type_cs[:, :, None] * tf.exp(-1 * s_type_alphas[:, :, None] * (tf.norm(R[:, None, :] - coords, axis=-1))[:, None, :])
        s_type_out = s_type_cs[:, :, None] * tf.math.exp(-1 * s_type_alphas[:, :, None] * (tf.norm(R[:, None, :] - coords, axis=-1))[:, None, :])
        s_type_out = tf.math.reduce_sum(s_type_out, axis=1)
        s_type_out = tf.math.unsorted_segment_sum(s_type_out, mol_ids, n_mol)
        # create the contribution from the p-type orbitals
        v_coeffs = self.reshape(v_coeffs)
        p_type_out = p_type_cs[:, :, None] * tf.exp(-1 * p_type_alphas[:, :, None] * (tf.norm(R[:, None, :] - coords, axis=-1))[:, None, :] ** 2)
        test = tf.math.reduce_sum((v_coeffs[:, :, None, : ] * tf.math.abs((R[:, None, :] - coords)[:, None, :, :])), axis=-1)
        #test = tf.math.reduce_sum((tf.constant([1., 0., 0.]) * tf.math.abs((R[:, None, :] - coords)[:, None, :, :])), axis=-1)
        p_type_out = p_type_out * test
        p_type_out = tf.math.reduce_sum(p_type_out, axis=1)
        p_type_out = tf.math.unsorted_segment_sum(p_type_out, mol_ids, n_mol)
        # normalizing 
        s_type_norms = tf.math.unsorted_segment_sum(s_type_integrals, mol_ids, n_mol)
        s_type_norms = tf.math.reduce_sum(s_type_norms, axis=-1)
        p_type_norms = tf.math.unsorted_segment_sum(p_type_integrals, mol_ids, n_mol)
        p_type_norms = tf.math.reduce_sum(p_type_norms, axis=-1)
        norms_sum = (s_type_norms + p_type_norms)[:, None]
        #dens_out = (s_type_out + p_type_out) / (s_type_norms + p_type_norms) * tf.cast(N_electrons, dtype=tf.float32)
        #dens_out = (s_type_out + p_type_out) #/ norms_sum * N_electrons
        dens_out = s_type_out + p_type_out #/ norms_sum
        return dens_out