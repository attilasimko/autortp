import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Reshape
tf.keras.mixed_precision.set_global_policy('mixed_float16')
import numpy as np
from utils import get_monaco_projections


def build_model(batch_size=1, num_cp=6):
    inp = Input(shape=(64, 64, 64, 2), dtype=tf.float16, batch_size=batch_size)
    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same')(inp)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(x)
    latent_space = Flatten()(x)


    dose = monaco_plan(inp[..., 0], latent_space, num_cp)

    return Model(inp, dose)
    
    

def monaco_plan(ct, latent_space, num_cp):
    control_matrices = get_monaco_projections(num_cp)
    # control_matrices = [tf.Variable(tf.random.normal(shape=(ct.shape[0], 64, 64, 64), dtype=tf.float16), trainable=False, dtype=tf.float16) for _ in range(240)]

    dose = tf.zeros_like(ct)
    leafs = []
    for i in range(num_cp):
        leaf = Dense(128, activation='relu')(latent_space)
        leaf = Dense(128, activation='relu')(leaf)
        leafs.append(Reshape((2, 64), name=f'leaf_{i}')(leaf))
    
    mus = []
    for i in range(num_cp):
        mu = Dense(32, activation='relu')(latent_space)
        mu = Dense(8, activation='relu')(mu)
        mu = Dense(4, activation='relu')(mu)
        mu = Dense(1, activation='relu')(mu)
        mus.append(mu)

    for cp_idx in range(num_cp):
        for j in range(64): # Number of leaves
            i_values = tf.range(64, dtype=tf.float32)
            i_values = tf.cast(i_values, tf.float16)
            i_values = tf.expand_dims(i_values, axis=-1)  # Shape: (64, 1)
            leaf_min = tf.expand_dims(leafs[cp_idx][:, 0, j], axis=-1)  # Shape: (batch_size, 1)
            leaf_max = tf.expand_dims(leafs[cp_idx][:, 1, j], axis=-1)  # Shape: (batch_size, 1)
            idx_value = j * 64 + i_values + 1
            
            condition = tf.logical_and(tf.less_equal(i_values, leaf_max), tf.greater_equal(i_values, leaf_min))
            dose += tf.where(condition, tf.where(tf.equal(control_matrices[cp_idx], idx_value), mus[cp_idx], 0), 0)
            
    return dose