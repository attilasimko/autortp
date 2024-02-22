from utils import get_dose_batch, vector_to_param, get_monaco_projections
import tensorflow as tf

def rtp_loss(num_cp):
    control_matrices = get_monaco_projections(num_cp)
    def loss_fn(y_true, y_pred):
        leafs, mus = vector_to_param(y_pred)
        dose = get_dose_batch(control_matrices, leafs, mus, y_true.shape)
        return tf.losses.mean_squared_error(y_true, dose)
    return loss_fn