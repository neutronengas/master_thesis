import tensorflow as tf
import json

def spatial_multipliers(Z, R, coords):

    # hard_coded based on the basis used, in this case cc-pvdz
    # shape is defined by (n_atoms, 2, max_n_angmom=3, max_n_orbitals_per_angmom=5, max_number_coeffs_per_ao=12)
    max_shape = tf.TensorShape((1, 2, 3, 5, 12))

    
    # R has shape (n_atoms, 3), coords has shape (n_atoms, n_coords, 3)
    R_min_coords = R[:, None, :] - coords
    # transpose the spatial dimension to be the tensor's first, to ease assignment
    R_min_coords_tranposed = tf.transpose(R_min_coords, (2, 0, 1))

    (n_atoms, n_coords, _) = R_min_coords.shape

    # the spatial multipliers for the 3 p-orbitals are given by x, y, z
    [x, y, z] = R_min_coords_tranposed
    

    # the spatial multipliers for the 5 d-orbitals are given by xy, xz, yz, x^2 - y^2, 3z^2 - r^2 (equals 2z^2 - 3x^2 - 3y^2)
    [xy, xz, yz, x2_min_y2, threez2_min_r2] = [
        x * y,
        x * z,
        y * z,
        x ** 2 - y ** 2,
        2 * z ** 2 - 3 * x ** 2 - 3 * y ** 2
    ]

    # the s-spatial multiplier is given by 1, with padded dimensions with shape (n_atoms, n_coords, max_n_orbitals_per_angmom=5)
    s_spatial_multiplier = tf.ones((n_atoms, n_coords, 5))
    
    # the p-spatial multiplier happens to be given by R_min_coords itself, with dimensions padded
    p_spatial_multiplier = R_min_coords
    p_spatial_multiplier = tf.pad(p_spatial_multiplier, paddings=[[0, 0], [0, 0], [0, 2]])

    # the d-spatial multiplier is given by the above defined variables
    d_spatial_multiplier = tf.concat([xy[None, :], xz[None, :], yz[None, :], x2_min_y2[None, :], threez2_min_r2[None, :]], axis=0)
    d_spatial_multiplier = tf.transpose(d_spatial_multiplier, (1, 2, 0))

    # concat all orbital multipliers to shape (3, n_atoms, n_coords, mmax_n_orbitals_per_angmom=5)
    all_orbital_multiplier = tf.concat([s_spatial_multiplier[None, :], p_spatial_multiplier[None, :], d_spatial_multiplier[None, :]], axis=0)
    # transpose to shape (n_atoms, n_coords, max_n_angmom=3, max_n_orbitals_per_angmom=5)
    all_orbital_multiplier = tf.transpose(all_orbital_multiplier, (1, 2, 0, 3))


    # the exponent mutiplier is a ones-tensor of shape (n_atoms, n_coords, max_n_angmom=3, max_n_orbitals_per_angmom=5)
    exponent_multiplier = tf.ones(all_orbital_multiplier.shape)

    # total multiplier, with shape (n_atoms, n_coords, 2, m_max=3, max_no_orbitals_per_m=4, max_split_per_m=5, max_number_coeffs_per_ao=12)
    multiplier = tf.concat([all_orbital_multiplier[:, :, None, :, :], exponent_multiplier[:, :, None, :, :]], axis=2)
    multiplier = multiplier[:, :, :, :, None, :, None]
    return multiplier