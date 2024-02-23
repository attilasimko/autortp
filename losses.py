from utils import get_dose_batch, vector_to_monaco_param, get_monaco_projections, get_absorption_matrices, get_dose_value
import tensorflow as tf

def rtp_loss(num_cp, num_mc):
    ray_matrices = get_monaco_projections(num_cp)
    def loss_fn(y_true, y_pred):
        absorption_matrices = get_absorption_matrices(y_true, num_cp)
        leafs, mus = vector_to_monaco_param(y_pred)
        dose_diffs = 0.0
        for mc_idx in range(num_mc):
            mc_point = tf.random.uniform([3], 0, 64, dtype=tf.int32)
            true_dose = y_true[:, mc_point[0], mc_point[1], mc_point[2]]
            pred_dose = get_dose_value(num_cp, absorption_matrices, ray_matrices, leafs, mus, mc_point)

            dose_diffs += tf.abs(true_dose - pred_dose) / num_mc

        return dose_diffs
    return loss_fn