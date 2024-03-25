import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, AveragePooling2D, MaxPooling3D, MaxPooling2D, Concatenate, Flatten, Dense, UpSampling2D, Dropout, Reshape, Activation, Conv1D, Conv2D
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

def build_model(batch_size, img_shape, decoder_info):
    inp = Input(shape=img_shape[1:], batch_size=batch_size)

    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inp[..., 0:2])
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((2, 2, 4))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = x[:, :, :, 0, :]
    latent_space = x

    if decoder_info[0] == "monaco":
        _, num_cp, dose_shape, img_shape, num_slices, leaf_length = decoder_info
        monaco = MonacoDecoder(num_cp, dose_shape, img_shape, num_slices, leaf_length)
        dose = monaco.predict_dose(latent_space, inp)
    else:
        raise ValueError("Decoder not implemented")

    return Model(inp, tf.concat([dose, inp[..., 1:]], 4))

class MonacoDecoder():
    def __init__(self, num_cp, dose_shape, img_shape, num_slices, leaf_length):
        self.img_shape = img_shape
        self.dose_shape = dose_shape
        self.num_slices = num_slices
        self.leaf_length = leaf_length
        self.num_cp = num_cp
        self.leaf_alpha = 0.001
        self.mu_alpha = 0.001
        self.mu_epsilon = 0.1
        self.ray_matrices = self.get_monaco_projections()

    def predict_dose(self, latent_vector, inp):
        ct = inp[..., 0:1]
        structs = inp[..., 1:]

        absorption_matrices = self.get_absorption_matrices(ct[..., 0:1])

        x = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer="he_normal")(latent_vector)
        img_upsampling = tf.minimum(4, self.num_slices // x.shape[2])
        leaf_upsampling = tf.minimum(4, self.leaf_length // x.shape[1])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        img_upsampling = tf.minimum(4, self.num_slices // x.shape[2])
        leaf_upsampling = tf.minimum(4, self.leaf_length // x.shape[1])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        img_upsampling = tf.minimum(4, self.num_slices // x.shape[2])
        leaf_upsampling = tf.minimum(4, self.leaf_length // x.shape[1])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        img_upsampling = tf.minimum(4, self.num_slices // x.shape[2])
        leaf_upsampling = tf.minimum(4, self.leaf_length // x.shape[1])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(self.num_cp, 1, activation='sigmoid', padding='same', kernel_initializer="he_normal")(x)
        leaf_total = Conv2D(self.num_cp, 1, activation='sigmoid', padding='same', kernel_initializer="he_normal")(x)
        leaf_lower, leaf_upper = tf.split(leaf_total, 2, axis=1)
        leaf_lower = tf.math.cumprod(leaf_lower, axis=1)
        leaf_upper = tf.math.cumprod(leaf_upper, axis=1, reverse=True)
        leaf_total = Concatenate(name="mlc", axis=1)([leaf_upper, leaf_lower])
        

        mu_total = Conv2D(self.num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        img_downsampling = tf.minimum(4, mu_total.shape[1])
        leaf_downsampling = tf.minimum(4, mu_total.shape[2])
        mu_total = MaxPooling2D(pool_size=(img_downsampling.numpy(), leaf_downsampling.numpy()))(mu_total)
        mu_total = Conv2D(self.num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(mu_total)
        img_downsampling = tf.minimum(4, mu_total.shape[1])
        leaf_downsampling = tf.minimum(4, mu_total.shape[2])
        mu_total = MaxPooling2D(pool_size=(img_downsampling.numpy(), leaf_downsampling.numpy()))(mu_total)
        mu_total = Conv2D(self.num_cp, 3, activation='relu', padding='same', kernel_initializer="he_normal")(mu_total)
        img_downsampling = tf.minimum(4, mu_total.shape[1])
        leaf_downsampling = tf.minimum(4, mu_total.shape[2])
        mu_total = MaxPooling2D(pool_size=(img_downsampling.numpy(), leaf_downsampling.numpy()))(mu_total)
        mu_total = Flatten()(mu_total)
        mu_total = Dense(4 * self.num_cp, activation='relu', kernel_initializer="he_normal")(mu_total)
        mu_total = Dense(2 * self.num_cp, activation='relu', kernel_initializer="he_normal")(mu_total)
        mu_total = Dense(self.num_cp, activation='relu', kernel_initializer="he_normal", name="mus")(mu_total) # , activity_regularizer=mu_reg(mu_alpha)
        
        ray_strengths = self.get_rays(absorption_matrices, leaf_total, mu_total)
        
        return tf.expand_dims(tf.reduce_sum(ray_strengths, 0), -1)

    def get_rays(self, absorption_matrices, leafs, mus):
        ray_strengths = []
        for cp_idx in range(len(self.ray_matrices)):
            batch_rays = []
            for batch_idx in range(leafs.shape[0]):
                ray_slices = []
                for slice_idx in range(leafs.shape[2]):
                    current_leafs = leafs[batch_idx, ..., slice_idx, cp_idx]
                    ray_matrix = tf.cast(self.ray_matrices[cp_idx][batch_idx, ...], dtype=tf.int32)
                    current_rays = tf.gather(current_leafs, ray_matrix)
                    current_rays = tf.expand_dims(tf.expand_dims(current_rays, 0), 3)
                    current_rays = UpSampling2D((1, self.dose_shape[1] // leafs.shape[1]))(current_rays)
                    ray_slices.append(current_rays[0, ..., 0])
                ray_stack = UpSampling2D((self.img_shape[0] // self.dose_shape[0]))(tf.expand_dims(tf.stack(ray_slices, -1), 0))[0, ...]
                absorbed_rays = tf.expand_dims(tf.multiply(ray_stack, absorption_matrices[batch_idx][:, :, :, cp_idx]), 0)
                batch_rays.append(tf.multiply(absorbed_rays, mus[batch_idx, cp_idx] + self.mu_epsilon))
            ray_strengths.append(Concatenate(0, name=f"ray_{cp_idx}")(batch_rays))
        return ray_strengths

    def get_absorption_matrices(self, ct):
        batches = []
        absorption_scalar = 1 / (96)
        absorption = tf.stop_gradient(tf.identity(tf.ones(ct.shape) * absorption_scalar))
        for batch in range(ct.shape[0]):
            rotated_arrays = []
            for idx in range(self.num_cp):
                angle = idx * 360 / self.num_cp
                array = absorption[batch, ...]
                array = tfa.image.rotate(array, angle, fill_mode='nearest', interpolation='bilinear')
                array = 1 - tf.cumsum(array, axis=0)
                array = tfa.image.rotate(array, - angle, fill_mode='nearest', interpolation='bilinear')
                array = tf.where(tf.greater(array, 0), array, 0)
                array = tf.where(tf.greater(ct[batch, ...], -0.8), array, 0)
                array /= tf.reduce_max(array)
                rotated_arrays.append(array)
            const_array = tf.cast(tf.concat(rotated_arrays, axis=-1), dtype=tf.float32)
            batches.append(const_array)
        
        return batches

    def get_monaco_projections(self):
        rotated_arrays = []
        indeces, _ = np.meshgrid(np.arange(self.leaf_length), np.arange(self.dose_shape[0]))

        for angle_idx in range(self.num_cp):
            array = np.expand_dims(rotate(indeces, - angle_idx * 360 / self.num_cp, reshape=False, order=0, mode='nearest'), 0)
            rotated_arrays.append(array)

        return [tf.constant(x) for x in rotated_arrays]