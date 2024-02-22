from utils import get_dose_batch, vector_to_param
import tensorflow as tf

def rtp_loss(y_true, y_pred):
    leafs, mus = vector_to_param(y_pred)
    dose = get_dose_batch(leafs, mus, y_true.shape)
    return tf.losses.mean_squared_error(y_true, dose)