from utils import *
import tensorflow as tf

def rtp_loss(num_cp, num_mc, leaf_length):
    ray_matrices = get_monaco_projections(num_cp)
    def loss_fn(y_true, y_pred):
        absorption_matrices = get_absorption_matrices(y_true, num_cp)
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp)
        dose_diffs = tf.cast([0.0], tf.float16)
        for batch_idx in range(y_true.shape[0]):
            dose_values = tf.cast(tf.unique(tf.reshape(y_true[batch_idx, ...], [-1])), tf.float16)
            for dose_value in dose_values:
                boolean_array = tf.equal(y_true[batch_idx, ...], dose_value)
                true_indices = tf.where(boolean_array)
                for mc_idx in range(tf.minimum(num_mc, tf.size(true_indices))):
                    mc_point = tf.random.shuffle(true_indices)[mc_idx]
                    # mc_point = tf.random.uniform([3], 0, 64, dtype=tf.int32)
                    true_dose = y_true[:, mc_point[0], mc_point[1], mc_point[2]]
                    pred_dose = get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point)
                    dose_diffs += tf.cast(tf.abs(dose_value - pred_dose) / num_mc, tf.float16)

        return dose_diffs
    return loss_fn