from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
import tensorflow as tf
def rtp_loss(weights):
    def loss_fn(y_true, y_pred):
        if (y_pred.shape[4] != len(weights) + 1):
            raise ValueError("Number of segmentations does not match number of weights")
        
        loss = mean_absolute_error(y_true, y_pred[..., 0:1])
        for i in range(len(weights)):
            loss = tf.where(tf.greater(y_pred[..., i+1], 0.5), loss * weights[i], loss)
        return loss
    return loss_fn