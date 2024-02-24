import numpy as np
from scipy.ndimage import rotate
import tensorflow as tf

def get_absorption_matrices(ct, num_cp):
    rotated_arrays = []
    for angle_idx in range(num_cp):
        rotated_arrays.append(tf.ones_like(ct, dtype=np.float16))

    return rotated_arrays

def get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point):
    dose = 0.0
    for cp_idx in range(len(ray_matrices)):
        absorption_value = absorption_matrices[cp_idx][:, mc_point[0], mc_point[1], mc_point[2]]
        leaf_idx = mc_point[2]
        ray_idx = tf.cast(ray_matrices[cp_idx][:, mc_point[0], mc_point[1], mc_point[2]], dtype=tf.float16)
        ray_idx -= tf.cast(64 * leaf_idx, dtype=tf.float16)
        cond_leafs = tf.logical_and(tf.greater_equal(ray_idx, leafs[:, 0, leaf_idx, cp_idx] * 32), tf.less_equal(ray_idx, 64 - leafs[:, 1, leaf_idx, cp_idx] * 32))
        dep_value = mus[:, 0, cp_idx] * absorption_value
        dose += tf.reduce_sum(tf.where(cond_leafs, dep_value, 0))
    return dose

def get_monaco_projections(num_cp):
    shape = (64, 64)
    rotated_arrays = []
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indeces = y * shape[0] + x
    indeces = np.repeat(indeces[..., None], 64, -1)
    indeces = np.swapaxes(indeces, 0, 2)

    for angle_idx in range(num_cp):
        array = np.expand_dims(rotate(indeces, angle_idx * 360 / num_cp, (0, 1), reshape=False, order=0, cval=-1), 0)
        rotated_arrays.append(array)

    return [tf.constant(x, dtype=tf.float16) for x in rotated_arrays]

def get_dose_batch(control_matrices, leafs, mus, dose_shape, num_cp=6):
    dose = tf.zeros(dose_shape, dtype=tf.float16)
    for cp_idx in range(num_cp):
        for j in range(64): # Number of leaves
            leaf_min = tf.expand_dims(leafs[:, 0, j, cp_idx], axis=-1)  # Shape: (batch_size, 1)
            leaf_max = tf.expand_dims(leafs[:, 1, j, cp_idx], axis=-1)  # Shape: (batch_size, 1)
            for i in range(64): # Number of leaves
                idx_value = j * 64 + i + 1
                leaf_idx_pos = tf.cast(i / 64, dtype=tf.float16)
                
                condition = tf.logical_and(tf.logical_and(tf.less_equal(leaf_idx_pos, leaf_max), tf.greater_equal(leaf_idx_pos, leaf_min)), tf.equal(control_matrices[cp_idx], idx_value))
                dose += tf.where(condition, mus[..., cp_idx], 0)
    return dose

def monaco_param_to_vector(leafs, mus):
    leafs = tf.stack(leafs, -1)
    mus = tf.stack(mus, -1)

    return tf.concat([tf.reshape(leafs, [-1]), tf.reshape(mus, [-1])], -1)

def vector_to_monaco_param(vec, leaf_length=64, num_cp=6):
    leafs = tf.reshape(vec[:leaf_length * 2 * num_cp], [1, 2, leaf_length, num_cp])
    mus = tf.reshape(vec[leaf_length * 2 * num_cp:], [1, 1, num_cp])
    return leafs, mus