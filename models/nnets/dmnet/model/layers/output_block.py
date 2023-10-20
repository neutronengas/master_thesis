import numpy as np
import tensorflow as tf
import json
import math
from utils.extract_coeffs import extract_coeffs
from utils.spatial_multipliers import spatial_multipliers

from tensorflow.keras import layers

class OutputBlock(layers.Layer):
    def __init__(self, num_grid_points, m_max, max_no_orbitals_per_m, max_split_per_m, max_number_coeffs_per_ao, emb_size, n_coords, name='output', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_grid_points = num_grid_points
        self.emb_size = emb_size
        self.m_max = m_max
        self.max_no_orbitals_per_m = max_no_orbitals_per_m
        self.max_split_per_m = max_split_per_m
        self.max_number_coeffs_per_ao = max_number_coeffs_per_ao
        self.n_coords = n_coords

        # total number of orbitals per atom
        no_orbitals_per_atom = m_max * max_no_orbitals_per_m * max_split_per_m

        # create vector of elements from which later DMs are created
        self.dense = tf.keras.layers.Dense((no_orbitals_per_atom) * (no_orbitals_per_atom - 1) / 2, activation=None)


    def call(self, inputs):
        
        # out: (n_atom_pairs, self.emb_size); Z: (n_atoms,); R: (n_atoms, 3); coords: (n_molecule, self.num_grid_points, 3), N: (n_molecule,)
        out, Z, R, coords, N, atom_pair_indices, atom_pair_mol_id = inputs
        
        # create guess for the density matrix 
        out = self.dense(out)
        dm_guess = self.reshape_dm(out)

        # partition DM values into molecules, by first creating the splitting indices
        row_limits = tf.math.cumsum(N)
        dm_guess = tf.RaggedTensor.from_row_limits(dm_guess, row_limits=row_limits, )

        # number of molecules within batch
        n_mol = tf.shape(N)[0]
        
        # pairwise euclidean distance between the atomic positions and coordinates of density evaluation (n_atoms, n_coords)
        R_min_coords = tf.norm(R[:, None, :] - coords, axis=-1)
        
        # recreate grid points for each atom of the molecule separately
        coords = tf.repeat(coords, N, axis=0)
        
        # get all contraction coefficients and exponents, without orbital-type-dependent prefactor
        coeffs = extract_coeffs(Z)
        
        # get orbital-type-dependent prefactors
        multipliers = spatial_multipliers(Z, R, coords)
        
        # integrate the p- and d-type orbital prefactors into the contraction coefficients by multiplying with prefactors
        coeffs = coeffs * multipliers
        
        # split into contraction coefficients and exponents
        coefficients = coeffs[:, :, None]
        exponents = coeffs[:, :, None]

        # create orbital values
        # orbitals are of shape 
        orbitals = coefficients * tf.math.exp(-1 * exponents * R_min_coords[:, :, None, None, None, None])

        # sum over the contraction coefficients to create atomic GTOs
        orbitals = tf.math.reduce_sum(orbitals, axis=-1)

        # flatten all orbital-dimensions, s.t. the final shape is given by (n_atoms, n_coords, no_orbitals_per_atom)
        orbitals = self.reshape_orbitals(orbitals)

        # transpose n_coords and no_orbitals_per_atom to make future assignements easier 
        # to shape (n_atoms, no_orbitals_per_atom, n_coords)
        orbitals = tf.transpose(orbitals, (0, 2, 1))

        # atom-pair indices are of shape (n_pairs, 2)
        # atom_pairs are of shape (n_pairs, 2, no_orbitals_per_atom, n_coords)
        atom_pairs = tf.gather(orbitals, atom_pair_indices)
    
        # multiply together the 2 orbitals
        orbital_pairs_multiplied = tf.math.reduce_prod(atom_pairs, axis=1)

        # multiply the orbital values with the DM values
        orbital_pairs_multiplied = orbital_pairs_multiplied * out

        # sum over the atom-pairs, grouped by the molecule to which they belong
        # output shape: (n_mol, n_coords)
        densities_molecule_wise = tf.math.unsorted_segment_sum(orbital_pairs_multiplied, atom_pair_mol_id, num_segments=n_mol)
        
        return densities_molecule_wise