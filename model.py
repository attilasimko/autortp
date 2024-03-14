import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Reshape, Activation, Conv1D
import numpy as np
from utils import get_monaco_projections, monaco_param_to_vector


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
    inp = Input(shape=(64, 64, 64, 2), dtype=tf.float16, batch_size=batch_size)
    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inp)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(2 * 128, (3, 3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Reshape((64, -1))(x)
    x = Conv1D(8, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv1D(16, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv1D(32, 3, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv1D(2 * num_cp, 12, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv1D(2 * num_cp, 12, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    x = Conv1D(2 * num_cp, 12, activation='relu', padding='same', kernel_initializer="he_normal")(x)
    latent_space = x

    concat = monaco_plan(latent_space, num_cp)

    return Model(inp, concat)
    
    

def monaco_plan(latent_space, num_cp):
    leaf_alpha = 0.0
    mu_alpha = 0.0

    leafs = []
    leaf_total = Conv1D(2 * num_cp, 12, activation='sigmoid', padding='same', kernel_initializer="he_normal")(latent_space)
    leaf_total = tf.split(leaf_total, 2*num_cp, axis=2)
    for i in range(num_cp):
        leafs.append(tf.concat([leaf_total[i * 2], leaf_total[i * 2 + 1]], 2, name=f'leaf_{i}'))
    
    mus = []
    mu_total = Conv1D(24, 3, activation='relu', padding='same', kernel_initializer="he_normal")(latent_space)
    mu_total = Conv1D(4, 3, activation='relu', padding='same', kernel_initializer="he_normal")(mu_total)
    mu_total = Flatten()(mu_total)
    mu_total = Dense(4 * num_cp, activation='relu', kernel_initializer="he_normal")(mu_total)
    mu_total = Dense(2 * num_cp, activation='relu', kernel_initializer="he_normal")(mu_total)
    mu_total = Dense(num_cp, activation='sigmoid', kernel_initializer="he_normal")(mu_total)
    mu_total = tf.split(mu_total, num_cp, axis=1)
    for i in range(num_cp):
        mus.append(Reshape((), name=f"mu_{i}")(mu_total[i]))
            
    return monaco_param_to_vector(leafs, mus)