import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, UpSampling2D, Dropout, Reshape, Activation, Conv1D, Conv2D
import numpy as np
from scipy.ndimage import rotate
import tensorflow_addons as tfa
tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()


class leaf_reg(keras.regularizers.Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * tf.math.reduce_mean(tf.math.abs(1 - weights))
    
class mu_reg(keras.regularizers.Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * tf.math.reduce_mean(tf.math.abs(weights))
    
        

def build_model(batch_size=1, num_cp=6):
    inp = Input(shape=(64, 64, 64, 2), batch_size=batch_size)
    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inp)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(2 * 128, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((2, 2, 4))(x)
    x = x[:, :, :, 0, :]
    x = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(4 * num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv2D(2 * num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv2D(num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)

    x = MonacoLayer(x, inp, num_cp)

    return Model(inp, x)

def MonacoLayer(latent_vector, ct, num_cp):
    ray_matrices = get_monaco_projections(num_cp)
    absorption_matrices = get_absorption_matrices(ct[..., 0:1], num_cp)
    leaf_alpha = 0.001
    mu_alpha = 0.001

    leaf_total = Conv2D(num_cp, 1, activation='sigmoid', padding='same', kernel_initializer="he_normal")(latent_vector)
    leaf_lower, leaf_upper = tf.split(leaf_total, 2, axis=1)
    leaf_lower = tf.math.cumprod(leaf_lower, axis=1)
    leaf_upper = tf.math.cumprod(leaf_upper, axis=1, reverse=True)
    leaf_total = tf.concat([leaf_upper, leaf_lower], 1)
    
    ray_strengths = get_rays(ray_matrices, absorption_matrices, leaf_total)

    # mus = []
    # mu_total = Conv2D(2 * num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(latent_vector)
    # mu_total = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer="he_normal")(mu_total)
    # mu_total = Flatten()(mu_total)
    # mu_total = Dense(4 * num_cp, activation='relu', kernel_initializer="he_normal")(mu_total)
    # mu_total = Dense(2 * num_cp, activation='relu', kernel_initializer="he_normal")(mu_total)
    # mu_total = Dense(num_cp, activation='sigmoid', kernel_initializer="zeros")(mu_total) # , activity_regularizer=mu_reg(mu_alpha)
    # mu_total = tf.split(mu_total, num_cp, axis=1)
    # for i in range(num_cp):
    #     mus.append(Reshape((), name=f"mu_{i}")(mu_total[i]))
    
    
    
    return tf.reduce_sum(ray_strengths, 0)

def get_rays(ray_matrices, absorption_matrices, leafs):
    ray_strengths = []
    for batch_idx in range(leafs.shape[0]):
        batch_rays = []
        for cp_idx in range(len(ray_matrices)):
            ray_slices = []
            for slice_idx in range(leafs.shape[2]):
                current_leafs = leafs[batch_idx, ..., slice_idx, cp_idx]
                ray_matrix = tf.cast(ray_matrices[cp_idx][batch_idx, ...], dtype=tf.int32)
                ray_slices.append(tf.gather(current_leafs, ray_matrix))
            ray_stack = tf.stack(ray_slices, -1)
            absorbed_rays = tf.multiply(ray_stack, absorption_matrices[batch_idx][:, :, :, cp_idx], name=f"rays_{cp_idx}")
            batch_rays.append(absorbed_rays)
        ray_strengths.append(tf.stack(batch_rays, 0))
    return ray_strengths

def get_absorption_matrices(ct, num_cp):
    batches = []
    absorption_scalar = 1 / (96)
    absorption = tf.identity(tf.ones(ct.shape) * absorption_scalar)
    for batch in range(ct.shape[0]):
        rotated_arrays = []
        for idx in range(num_cp):
            array = absorption[batch, ...]
            array = tfa.image.rotate(array, idx * 360 / num_cp, fill_mode='nearest', interpolation='bilinear')
            array = 1 - tf.cumsum(array, axis=0)
            array = tfa.image.rotate(array, - idx * 360 / num_cp, fill_mode='nearest', interpolation='bilinear')
            array = tf.where(tf.greater(array, 0), array, 0)
            array = tf.where(tf.greater(ct[batch, ...], -0.8), array, 0)
            array /= tf.reduce_max(array)
            rotated_arrays.append(array)
        const_array = tf.cast(tf.concat(rotated_arrays, axis=-1), dtype=tf.float32)
        batches.append(const_array)
    
    return batches

def get_monaco_projections(num_cp):
    shape = (64, 64)
    rotated_arrays = []
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indeces = x

    for angle_idx in range(num_cp):
        array = np.expand_dims(rotate(indeces, - angle_idx * 360 / num_cp, reshape=False, order=0, mode='nearest'), 0)
        rotated_arrays.append(array)

    return [tf.constant(x) for x in rotated_arrays]