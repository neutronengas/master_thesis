import tensorflow as tf
import numpy as np
import json


def extract_coeffs_elementwise(z):
    # relative path from location of executing notebook
    
    # test mode
    # data = open("./cc-pvdz.1.json").read()
    
    # use mode
    data = open("./utils/cc-pvdz.1.json").read()
    
    data = json.loads(data)
    # curtail unnecessary content
    data = data['elements'][str(int(z.numpy()))]['electron_shells']

    # number of angular momenta
    n_angmom = len(data)
    all_coefficients = []
    all_exponents = []
    for i in range(n_angmom):
        
        # number of orbitals per angular momentum: 1 for s, 3 for p, 5 for d
        # the spatial orbital multiplication factors are x, y, z for p and xy, xz, yz, x^2 - y^2, 3z^2 - r^2
        num_orbs = 2 * i + 1
        coefficients = data[i]['coefficients']

        # convert strings to floats
        coefficients = [list(map(float, coeffs)) for coeffs in coefficients]
        coefficients = np.array(coefficients, dtype=np.float32)
        exponents = data[i]['exponents']

        # convert strings to floats
        exponents = [float(exps) for exps in exponents]
        exponents = np.array(exponents, dtype=np.float32)

        # repeat exponents for each set of coefficients
        exponents = np.repeat(exponents[None, :], coefficients.shape[0], axis=0)

        # repeat coefficients / exponents based on number of orbitals per main quantum number: 1 for s, 3 for p, 5 for d
        coefficients = coefficients[:, None]
        coefficients = np.repeat(coefficients, num_orbs, axis=1)

        exponents = exponents[:, None]
        exponents = np.repeat(exponents, num_orbs, axis=1)


        all_coefficients.append(coefficients.tolist())
        all_exponents.append(exponents.tolist())
    
    tens_data = [all_coefficients, all_exponents]
    return tf.ragged.constant(tens_data)


def extract_coeffs(Z):
    ragged_tensor = tf.map_fn((extract_coeffs_elementwise), Z, fn_output_signature=tf.RaggedTensorSpec(shape=[2, None, None, None, None], dtype=tf.float32))
    out = ragged_tensor.to_tensor()[:, None]
    # out has shape (n_atoms, n_coords, 2, m_max=3, max_no_orbitals_per_m=4, max_split_per_m=5, max_coeff_per_ao=12)
    return out