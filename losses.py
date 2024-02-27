from utils import *
import tensorflow as tf

def rtp_loss(ray_matrices, num_cp, num_mc, leaf_length):
    def loss_fn(y_true, y_pred):
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp)
        dose_diffs = 0.0
        for batch_idx in range(y_true.shape[0]):
            absorption_matrices = get_absorption_matrices(y_true[batch_idx, ..., 0], num_cp)
            dose_values, _ = tf.unique(tf.reshape(y_true[batch_idx, ..., 0], [-1]))
            for dose_value in dose_values:
                dose_idxes = tf.where(tf.equal(y_true[batch_idx, ..., 0], dose_value))
                for mc_idx in range(tf.minimum(num_mc, tf.size(dose_idxes))):
                    mc_point = tf.random.shuffle(dose_idxes)[mc_idx]
                    pred_dose = get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point)
                    dose_diffs += tf.cast(tf.abs(tf.cast(dose_value, tf.float16) - pred_dose) / (num_mc * y_true.shape[0]), tf.float32)

        return dose_diffs
    return loss_fn