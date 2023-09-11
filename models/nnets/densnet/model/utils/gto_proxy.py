import numpy as np
import tensorflow as tf


def exponential_tensorwise(point_R_diff_transformed):
    # input has shape (len(Z), 3)
    return np.exp(-np.linalg.norm(point_R_diff_transformed, axis=1))
