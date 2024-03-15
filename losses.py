from utils import *
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def rtp_loss(ray_matrices, num_cp, grid_mc, leaf_length):
    def loss_fn(y_true, y_pred):
        num_eval_points = y_true.shape[0] * (64 // grid_mc) ** 3
        absorption_matrices = tf.split(y_true, y_true.shape[-1], axis=-1)[1:]
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp, y_true.shape[0])
        dose_diffs = []
        for batch_idx in range(y_true.shape[0]):
            for mc_x in range(tf.random.uniform([], 0, grid_mc, dtype=tf.int32), y_true.shape[1], grid_mc):
                for mc_y in range(tf.random.uniform([], 0, grid_mc, dtype=tf.int32), y_true.shape[2], grid_mc):
                    for mc_z in range(tf.random.uniform([], 0, grid_mc, dtype=tf.int32), y_true.shape[3], grid_mc):
                        mc_point = [mc_x, mc_y, mc_z]
                        pred_dose = get_dose_value([matrix[batch_idx, ...] for matrix in absorption_matrices], ray_matrices, leafs[batch_idx, ...], mus[batch_idx, ...], mc_point)
                        true_dose = y_true[batch_idx, mc_x, mc_y, mc_z, 0]
                        dose_diffs.append(tf.cast(tf.abs(tf.cast(true_dose, tf.float16) - pred_dose), tf.float32))
        return tf.reduce_mean(dose_diffs)
    return loss_fn