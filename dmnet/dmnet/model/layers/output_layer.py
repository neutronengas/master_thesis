import numpy as np
import tensorflow as tf
from ..utils.create_orbital_values import create_orbital_values
from datetime import datetime as dt
from ..layers.tensor_product_expansion import TensorProductExpansionLayer

from tensorflow.keras import layers
from ..layers.pairmix_layer import PairmixLayer

class OutputLayer(layers.Layer):
    def __init__(self, L, F, K, r_cut, cgc, atoms, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.F = F
        self.L = L
        self.atoms = atoms
        self.cgc = cgc
        self.initializer = tf.keras.initializers.GlorotNormal()

        self.pairmix_layer = PairmixLayer(L, L, L, F, K, r_cut, cgc)
        self.tens_prod_exp = TensorProductExpansionLayer(cgc)

    def build(self, shape):
        #self.reduce_feature_vec_matrix = self.add_weight(name="reduce_feature_vec_matrix", shape=(14, self.F,), dtype=tf.float32, trainable=True)
        # self.weight = self.add_weight(name="weight", shape=(len(self.atoms) ** 2, self.L, self.F, self.F), initializer=self.initializer)       
        pass

    def call(self, inputs):
        # out: (n_atoms, self.no_orbitals_per_atom, self.emb_size); Z: (n_atoms,); R: (n_atoms, 3); coords: (n_molecule, self.num_grid_points, 3), N: (n_molecule,)
        # atom_pair_indices: (n_pairs, 2), atom_pair_mol_id: (n_pairs,), rdm: (TODO), N_rdm: (TODO)
        atoms, pairs, Z, R, N, atom_pair_indices, atom_pair_mol_id, atom_idx, pair_idx, rdm, N_rdm = inputs
        n_atoms = len(atoms[0])
        n_pairs = len(pairs[0])
        l_dict = {
            (0, 0): {
                0: tf.range(0, 9) # (1s, 2s, 3s) x (1s, 2s, 3s) -> 9 pairs
            },
            (1, 0): {
                1: tf.range(0, 6) # (2p, 3p) x (1s, 2s, 3s) -> 6 pairs
            },
            (1, 1): {
                0: tf.range(9, 13), # (2p, 3p) x (2p, 3p) -> 4 pairs
                1: tf.range(6, 10),
                2: tf.range(0, 4)
            },
            (2, 0): {
                2: tf.range(4, 7) # (3d) x (1s, 2s, 3s) -> 3 pairs
            },
            (2, 1): {
                1: tf.range(10, 12), # (3d) x (2p, 3p) -> 2 pairs
                2: tf.range(7, 9),
                3: tf.range(0, 2)
            },
            (2, 2): {
                0: tf.range(13, 14), # (3d) x (3d) -> 1 pair
                1: tf.range(12, 13),
                2: tf.range(9, 10),
                3: tf.range(2, 3), 
                4: tf.range(0, 1) 
            }
        }
        
        # number of main quantum numbers per spin l; s = 0, p = 1, d = 2
        n_dict = {
            0: 3,
            1: 2,
            2: 1
        }

        M_atoms = {}
        M_pairs = {}
        for l1 in range(3):
            for l2 in range(l1 + 1):
                M_l1l2_atoms = tf.zeros((n_atoms, (2*l1+1)*n_dict[l1], (2*l2+1)*n_dict[l2]))
                M_l1l2_pairs = tf.zeros((n_pairs, (2*l1+1)*n_dict[l1], (2*l2+1)*n_dict[l2]))
                l3_range = [l1 - x for x in range(-l2, l2 + 1)]
                for l3 in l3_range:
                    self.tens_prod_exp.set_params(l1, l2, l3)
                    F_idx = l_dict[(l1, l2)][l3]

                    M_l1l2l3_atoms = self.tens_prod_exp([[tf.gather(atoms[l], F_idx, axis=1) for l in range(self.L + 1)]])
                    M_l1l2l3_atoms = tf.reshape(M_l1l2l3_atoms, (n_atoms, 2*l1+1, 2*l2+1, n_dict[l1], n_dict[l2]))
                    M_l1l2l3_atoms = tf.transpose(M_l1l2l3_atoms, (0, 1, 3, 2, 4))
                    M_l1l2l3_atoms = tf.reshape(M_l1l2l3_atoms, (n_atoms, (2*l1+1)*n_dict[l1], (2*l2+1)*n_dict[l2]))
                    M_l1l2_atoms += M_l1l2l3_atoms

                    M_l1l2l3_pairs = self.tens_prod_exp([[tf.gather(pairs[l], F_idx, axis=1) for l in range(self.L + 1)]])
                    M_l1l2l3_pairs = tf.reshape(M_l1l2l3_pairs, (n_pairs, 2*l1+1, 2*l2+1, n_dict[l1], n_dict[l2]))
                    M_l1l2l3_pairs = tf.transpose(M_l1l2l3_pairs, (0, 1, 3, 2, 4))
                    M_l1l2l3_pairs = tf.reshape(M_l1l2l3_pairs, (n_pairs, (2*l1+1)*n_dict[l1], (2*l2+1)*n_dict[l2]))
                    M_l1l2_pairs += M_l1l2l3_pairs
                   
                M_atoms[(l1, l2)] = M_l1l2_atoms
                M_pairs[(l1, l2)] = M_l1l2_pairs

        atoms_ss = M_atoms[(0, 0)]        
        atoms_ps = M_atoms[(1, 0)]
        atoms_ds = M_atoms[(2, 0)]
        atoms_xs = tf.concat([atoms_ss, atoms_ps, atoms_ds], axis=1)

        atoms_pp = M_atoms[(1, 1)]
        atoms_dp = M_atoms[(2, 1)]
        atoms_xp = tf.concat([tf.transpose(atoms_ps, (0, 2, 1)), 
                              atoms_pp, 
                              atoms_dp], axis=1)

        atoms_dd = M_atoms[(2, 2)]
        atoms_xd = tf.concat([tf.transpose(atoms_ds, (0, 2, 1)), 
                              tf.transpose(atoms_dp, (0, 2, 1)),
                              atoms_dd], axis=1)
        
        atoms = tf.concat([atoms_xs, atoms_xp, atoms_xd], axis=2)


        pairs_ss = M_pairs[(0, 0)]
        pairs_ps = M_pairs[(1, 0)]
        pairs_ds = M_pairs[(2, 0)]
        pairs_xs = tf.concat([pairs_ss, pairs_ps, pairs_ds], axis=1)

        pairs_pp = M_pairs[(1, 1)]
        pairs_dp = M_pairs[(2, 1)]
        pairs_xp = tf.concat([tf.transpose(pairs_ps, (0, 2, 1)), 
                              pairs_pp, 
                              pairs_dp], axis=1)

        pairs_dd = M_pairs[(2, 2)]
        pairs_xd = tf.concat([tf.transpose(pairs_ds, (0, 2, 1)), 
                              tf.transpose(pairs_dp, (0, 2, 1)),
                              pairs_dd], axis=1)
        
        pairs = tf.concat([pairs_xs, pairs_xp, pairs_xd], axis=2)

        rdm = tf.concat([atoms, pairs], axis=0)

        perm = tf.concat([atom_idx, pair_idx], axis=0)
        perm = tf.argsort(perm, axis=0)

        tf.debugging.assert_shapes([(tf.zeros((36, 14, 14)), ('N', 'F', 'M')), (rdm, ('N', 'F', 'M'))])
        rdm = tf.gather(rdm, perm)
        return rdm
    