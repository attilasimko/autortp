from utils import *
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def rtp_loss(ray_matrices, num_cp, num_mc, leaf_length):
    def loss_fn(y_true, y_pred):
        absorption_matrices = tf.split(y_true, y_true.shape[-1], axis=-1)[1:]
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp, y_true.shape[0])
        dose_diffs = 0.0
        for batch_idx in range(y_true.shape[0]):
            mc_center = [tf.random.uniform([], 0, y_true.shape[1], dtype=tf.int32), tf.random.uniform([], 0, y_true.shape[2], dtype=tf.int32)]
            for mc_z in range(y_true.shape[3]):
                mc_point = [mc_center[0], mc_center[1], mc_z]
                pred_dose = get_dose_value([matrix[batch_idx, ...] for matrix in absorption_matrices], ray_matrices, leafs[batch_idx, ...], mus[batch_idx, ...], mc_point)
                dose_diffs += tf.cast(tf.abs(tf.cast(y_true[batch_idx, mc_point[0], mc_point[1], mc_point[2], 0], tf.float16) - pred_dose) / (y_true.shape[0] * y_true.shape[3]), tf.float32)
                pred_dose = get_dose_value([matrix[batch_idx, ...] for matrix in absorption_matrices], ray_matrices, leafs[batch_idx, ...], mus[batch_idx, ...], mc_point)
                dose_value = y_true[batch_idx, mc_point[0], mc_point[1], mc_point[2], 0]
                dose_diffs += tf.cast(tf.abs(tf.cast(dose_value, tf.float16) - pred_dose) / (num_mc * y_true.shape[0]), tf.float32)
        return dose_diffs
    return loss_fn