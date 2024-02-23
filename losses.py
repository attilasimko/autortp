from utils import get_dose_batch, vector_to_monaco_param, get_monaco_projections, get_absorption_matrices
import tensorflow as tf

def rtp_loss(num_cp):
    ray_matrices = get_monaco_projections(num_cp)
    num_mc = 1
    def loss_fn(y_true, y_pred):
        absorption_matrices = get_absorption_matrices(y_true[..., 0], num_cp)
        leafs, mus = vector_to_monaco_param(y_pred)
        dose_diffs = 0.0
        for mc_idx in range(num_mc):
            dose = 0.0
            mc_point = tf.random.uniform([3], 0, 64, dtype=tf.int32)
            true_dose = y_true[mc_point[0], mc_point[1], mc_point[2]]
            for cp_idx in range(num_cp):
                absorption_value = absorption_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2]]
                print(ray_matrices[cp_idx])
                ray_idx = ray_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2]]
                cond_leafs = tf.logical_and(tf.less_equal(leafs[:, 1, mc_point[1], cp_idx], ray_idx), tf.greater_equal(leafs[:, 0, mc_point[1], cp_idx], ray_idx))
                dose += tf.reduce_sum(tf.where(cond_leafs, mus[:, 0, cp_idx], 0) * absorption_value)
            dose_diffs += tf.abs(true_dose - dose) / num_mc

        return dose_diffs
    return loss_fn