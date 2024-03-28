import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Add, Subtract, Multiply, AveragePooling2D, MaxPooling3D, UpSampling3D, MaxPooling2D, Concatenate, Flatten, Dense, UpSampling2D, Dropout, Reshape, Activation, Conv1D, Conv2D
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

def build_model(batch_size, img_shape, decoders):
    inp = Input(shape=(img_shape[0], img_shape[1], img_shape[2], 2), batch_size=batch_size)

    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inp)
    x = MaxPooling3D((4, 4, 5))(x)
    x = Conv3D(8, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((2, 2, 5))(x)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((2, 2, 4))(x)
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((2, 2, 1))(x)
    x = x[:, :, :, 0, :]
    latent_space = x

    models = []
    for decoder_info in decoders:
        _, num_cp, dose_shape, img_shape, num_slices, leaf_length = decoder_info
        monaco = MonacoDecoder(num_cp, dose_shape, img_shape, num_slices, leaf_length)
        dose = monaco.predict_dose(latent_space, inp)
        models.append(Model(inp, dose))

    return models

class MonacoDecoder():
    def __init__(self, num_cp, dose_resolution, img_shape, num_leafs, leaf_resolution):
        self.img_shape = img_shape
        self.dose_resolution = dose_resolution
        self.num_leafs = num_leafs
        self.leaf_resolution = leaf_resolution
        self.num_cp = num_cp
        self.leaf_alpha = 0.001
        self.mu_alpha = 0.001
        self.mu_epsilon = 0.1
        self.ray_matrices = self.get_monaco_projections()

    def predict_dose(self, latent_vector, inp):
        ct = inp[..., 0:1]
        structs = inp[..., 1:]


        x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer="he_normal")(latent_vector)
        leaf_upsampling = tf.minimum(4, self.leaf_resolution // x.shape[1])
        img_upsampling = tf.minimum(5, self.num_leafs // x.shape[2])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        leaf_upsampling = tf.minimum(4, self.leaf_resolution // x.shape[1])
        img_upsampling = tf.minimum(4, self.num_leafs // x.shape[2])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        leaf_upsampling = tf.minimum(4, self.leaf_resolution // x.shape[1])
        img_upsampling = tf.minimum(4, self.num_leafs // x.shape[2])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        leaf_upsampling = tf.minimum(4, self.leaf_resolution // x.shape[1])
        img_upsampling = tf.minimum(4, self.num_leafs // x.shape[2])
        x = UpSampling2D((leaf_upsampling, img_upsampling))(x)
        x = Conv2D(self.num_cp, 1, activation='sigmoid', padding='same', kernel_initializer="he_normal")(x)
        leaf_total = Conv2D(self.num_cp, 1, activation='sigmoid', padding='same', kernel_initializer="he_normal")(x)
        leaf_lower = tf.math.cumprod(leaf_total, axis=1)
        leaf_upper = tf.math.cumprod(leaf_total, axis=1, reverse=True)
        leaf_total = Multiply()([leaf_lower, leaf_upper])
        leaf_total = Subtract(name='mlc')([tf.ones_like(leaf_total), leaf_total])
        

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
        
        absorption_matrices = self.get_absorption_matrices(ct[..., 0:1], mu_total)

        dose = self.get_rays(absorption_matrices, leaf_total)
        
        return tf.expand_dims(dose, -1)

    def get_rays(self, absorption_matrices, leafs):
        ray_strengths = []
        for cp_idx in range(len(self.ray_matrices)):
            batch_rays = []
            for batch_idx in range(leafs.shape[0]):
                ray_slices = []
                for slice_idx in range(self.num_leafs):
                    current_leafs = leafs[batch_idx, ..., slice_idx, cp_idx]
                    ray_matrix = self.ray_matrices[cp_idx][batch_idx, ...]
                    current_rays = tf.gather(current_leafs, ray_matrix)
                    current_rays = tf.expand_dims(tf.expand_dims(current_rays, 0), 3)
                    ray_slices.append(current_rays[0, ..., 0])
                ray_stack = tf.expand_dims(tf.stack(ray_slices, -1), 0)
                ray_stack = UpSampling3D((self.img_shape[0] // self.dose_resolution, self.img_shape[1] // self.leaf_resolution, self.img_shape[2] // self.num_leafs))(ray_stack)[0, ...]
                absorbed_rays = tf.expand_dims(tf.multiply(ray_stack, absorption_matrices[batch_idx][:, :, :, cp_idx]), 0)
                batch_rays.append(absorbed_rays)
            ray_strengths.append(Concatenate(0, name=f"ray_{cp_idx}")(batch_rays))
        return tf.reduce_sum(ray_strengths, 0)

    def get_absorption_matrices(self, ct, mu_total):
        batches = []
        absorption = tf.identity(tf.ones(ct.shape, dtype=tf.float32))
        for batch in range(ct.shape[0]):
            rotated_arrays = []
            for idx in range(self.num_cp):
                angle = idx * 360 / self.num_cp
                array = absorption[batch, ...]
                array = tfa.image.rotate(array, angle, fill_mode='nearest', interpolation='bilinear')
                array = array.shape[0] - tf.cumsum(array, axis=0)
                array = tfa.image.rotate(array, - angle, fill_mode='nearest', interpolation='bilinear')
                array = tf.where(tf.greater(array, 0), array, 0)
                array = tf.where(tf.greater(ct[batch, ...], -0.8), array, 0)
                array /= tf.reduce_max(array)
                rotated_arrays.append(array)
            const_array = tf.stop_gradient(tf.cast(tf.concat(rotated_arrays, axis=-1), dtype=tf.float32))
            batches.append(tf.multiply(const_array, mu_total[batch, :][None, None, None, :]) + self.mu_epsilon)
        
        return batches

    def get_monaco_projections(self):
        rotated_arrays = []
        indeces, _ = np.meshgrid(np.arange(self.leaf_resolution), np.arange(self.dose_resolution))

        for angle_idx in range(self.num_cp):
            array = np.expand_dims(rotate(indeces, - angle_idx * 360 / self.num_cp, reshape=False, order=0, mode='nearest'), 0)
            rotated_arrays.append(array)

        return [tf.constant(x, dtype=tf.int32) for x in rotated_arrays]