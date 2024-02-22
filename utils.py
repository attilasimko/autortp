import numpy as np
from scipy.ndimage import rotate
import tensorflow as tf

def get_monaco_projections(num_cp):
    rotated_arrays = []
    shape = (128, 128, 128)
    new_shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    sigma = 10
    center = (64, 64)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

    indeces = y * shape[0] + x
    indeces = np.repeat(indeces[..., None], 128, -1)
    indeces = np.swapaxes(indeces, 1, 2)

    # d2 = (x - center[0])**2 + (y - center[1])**2
    # gaussian = np.exp(-d2 / (2 * sigma**2))
    # gaussian /= np.max(gaussian)
    # gaussian = np.repeat(gaussian[..., None], 128, -1)
    # gaussian = np.swapaxes(gaussian, 1, 2)

    for angle_idx in range(num_cp):
        array = np.expand_dims(rotate(indeces, angle_idx * 360 / num_cp, (0, 1), reshape=False, order=0)[32:96, 32:96, 32:96], 0)
        array_replica = - np.ones_like(array)
        idx = 0
        for i in range(32, 96):
            for j in range(32, 96):
                array_replica[array == (j * shape[0] + i)] = idx
                idx += 1
        rotated_arrays.append(np.array(array_replica, dtype=np.float16))

    return rotated_arrays

def get_dose_batch(leafs, mus, dose_shape, num_cp=6):
    dose = tf.zeros(dose_shape, dtype=tf.float16)
    control_matrices = get_monaco_projections(num_cp)
    for cp_idx in range(num_cp):
        for j in range(64): # Number of leaves
            i_values = tf.range(64, dtype=tf.float32)
            i_values = tf.cast(i_values, tf.float16)
            i_values = tf.expand_dims(i_values, axis=-1)  # Shape: (64, 1)
            leaf_min = tf.expand_dims(leafs[:, 0, j, cp_idx], axis=-1)  # Shape: (batch_size, 1)
            leaf_max = tf.expand_dims(leafs[:, 1, j, cp_idx], axis=-1)  # Shape: (batch_size, 1)
            idx_value = j * 64 + i_values + 1
            
            condition = tf.logical_and(tf.logical_and(tf.less_equal(i_values, leaf_max), tf.greater_equal(i_values, leaf_min)), tf.equal(control_matrices[cp_idx], idx_value))
            dose += tf.where(condition, mus[..., cp_idx], 0)
    return dose

def param_to_vector(leafs, mus):
    leafs = tf.stack(leafs, -1)
    mus = tf.stack(mus, -1)

    return tf.concat([tf.reshape(leafs, [-1]), tf.reshape(mus, [-1])], -1)

def vector_to_param(vec, num_cp=6):
    leafs = tf.reshape(vec[:64 * 2 * num_cp], [1, 2, 64, num_cp])
    mus = tf.reshape(vec[64 * 2 * 6:], [1, 1, num_cp])
    return leafs, mus