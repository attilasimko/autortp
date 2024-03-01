from utils import *
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def rtp_loss(ray_matrices, num_cp, num_mc, leaf_length):
    def loss_fn(y_true, y_pred):
        absorption_matrices = tf.split(y_true, y_true.shape[-1], axis=-1)[1:]
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp, y_true.shape[0])
        dose_diffs = 0.0
        for batch_idx in range(y_true.shape[0]):
            dose_values, _ = tf.unique(tf.reshape(y_true[batch_idx, ..., 0], [-1]))
            for dose_value in dose_values:
                dose_idxes = tf.where(tf.equal(y_true[batch_idx, ..., 0], dose_value))
                dose_idxes = tf.random.shuffle(dose_idxes)
                for mc_idx in range(tf.minimum(num_mc, tf.shape(dose_idxes)[0])):
                    mc_point = dose_idxes[mc_idx]
                    pred_dose = get_dose_value([matrix[batch_idx, ...] for matrix in absorption_matrices], ray_matrices, leafs[batch_idx, ...], mus[batch_idx, ...], mc_point)
                    dose_diffs += tf.cast(tf.abs(tf.cast(dose_value, tf.float16) - pred_dose) / (num_mc * y_true.shape[0]), tf.float32)
        return dose_diffs
    return loss_fn