from utils import *
import tensorflow as tf

def rtp_loss(num_cp, num_mc, leaf_length):
    ray_matrices = get_monaco_projections(num_cp)
    def loss_fn(y_true, y_pred):
        absorption_matrices = get_absorption_matrices(y_true, num_cp)
        leafs, mus = vector_to_monaco_param(y_pred, leaf_length, num_cp)
        dose_diffs = 0.0
        mc_points = [tf.constant([16, 16, 16], dtype=tf.int32),
                        tf.constant([16, 16, 32], dtype=tf.int32),
                        tf.constant([16, 16, 48], dtype=tf.int32),
                        tf.constant([16, 32, 16], dtype=tf.int32),
                        tf.constant([16, 32, 32], dtype=tf.int32),
                        tf.constant([16, 32, 48], dtype=tf.int32),
                        tf.constant([16, 48, 16], dtype=tf.int32),
                        tf.constant([16, 48, 32], dtype=tf.int32),
                        tf.constant([16, 48, 48], dtype=tf.int32),
                        tf.constant([32, 16, 16], dtype=tf.int32),
                        tf.constant([32, 16, 32], dtype=tf.int32),
                        tf.constant([32, 16, 48], dtype=tf.int32),
                        tf.constant([32, 32, 16], dtype=tf.int32),
                        tf.constant([32, 32, 32], dtype=tf.int32),
                        tf.constant([32, 32, 48], dtype=tf.int32),
                        tf.constant([32, 48, 16], dtype=tf.int32),
                        tf.constant([32, 48, 32], dtype=tf.int32),
                        tf.constant([32, 48, 48], dtype=tf.int32),
                        tf.constant([48, 16, 16], dtype=tf.int32),
                        tf.constant([48, 16, 32], dtype=tf.int32),
                        tf.constant([48, 16, 48], dtype=tf.int32),
                        tf.constant([48, 32, 16], dtype=tf.int32),
                        tf.constant([48, 32, 32], dtype=tf.int32),
                        tf.constant([48, 32, 48], dtype=tf.int32),
                        tf.constant([48, 48, 16], dtype=tf.int32),
                        tf.constant([48, 48, 32], dtype=tf.int32),
                        tf.constant([48, 48, 48], dtype=tf.int32)]
        for mc_idx in range(len(mc_points)):
            mc_point = mc_points[mc_idx] # tf.random.uniform([3], 0, 64, dtype=tf.int32)
            true_dose = y_true[:, mc_point[0], mc_point[1], mc_point[2]]
            pred_dose = get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point)

            dose_diffs += tf.abs(true_dose - pred_dose) / num_mc

        return dose_diffs
    return loss_fn