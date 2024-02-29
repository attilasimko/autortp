from utils import *
import tensorflow as tf

def rtp_loss(ray_matrices, num_cp, num_mc, leaf_length):
    def loss_fn(y_true, y_pred):
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp, y_true.shape[0])
        dose_diffs = 0.0
        for batch_idx in range(y_true.shape[0]):
            absorption_matrices = get_absorption_matrices(y_true[batch_idx, ..., 0], num_cp)
            dose_values, _ = tf.unique(tf.reshape(y_true[batch_idx, ..., 0], [-1]))
            for dose_value in dose_values:
                dose_idxes = tf.where(tf.equal(y_true[batch_idx, ..., 0], dose_value))
                dose_idxes = tf.random.shuffle(dose_idxes)
                for mc_idx in range(tf.minimum(num_mc, dose_idxes.shape[0])):
                    mc_point = dose_idxes[mc_idx]
                    pred_dose = get_dose_value(absorption_matrices, ray_matrices, leafs[batch_idx, ...], mus[batch_idx, ...], mc_point)
                    dose_diffs += tf.cast(tf.abs(tf.cast(dose_value, tf.float16) - pred_dose) / (num_mc * y_true.shape[0]), tf.float32)

        return dose_diffs
    return loss_fn