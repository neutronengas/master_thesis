import tensorflow as tf
import json

def spatial_multipliers(Z, R, coords):

    # test mode
    data = open("./utils/cc-pvdz.1.json").read()
    
    # use mode
    #data = open("./utils/cc-pvdz.1.json").read()
    
    data = json.loads(data)
    # curtail unnecessary content
    z = tf.reduce_max(Z).numpy()
    data = data['elements'][str(int(z))]['electron_shells']
    n_angmom = len(data)
    # maximum number of orbitals per angular momentum
    n_orbitals = 2 * n_angmom - 1

    coefficient_lengths = [[len(y) for y in shell['coefficients']] for shell in data]
    # coeff_ls is list of ints -> max(coeff_ls) is int -> num_coefficients is list of ints
    num_coefficients = [max(coeff_ls) for coeff_ls in coefficient_lengths]
    max_num_coefficients = max(num_coefficients)

    

    return tf.TensorShape((tf.shape(Z)[0], 2, n_angmom, n_orbitals, max_num_coefficients))