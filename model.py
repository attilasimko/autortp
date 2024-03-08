import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Reshape, Activation, Conv1D
import numpy as np
from utils import get_monaco_projections, monaco_param_to_vector


def build_model(batch_size=1, num_cp=6):
    inp = Input(shape=(64, 64, 64, 2), dtype=tf.float16, batch_size=batch_size)
    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same')(inp)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(2 * 128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Reshape((1, -1))(x)
    x = Conv1D(4 * 128, 3, activation='relu', padding='same')(x)
    x = Conv1D(8 * 128, 3, activation='relu', padding='same')(x)
    x = Conv1D(16 * 128, 3, activation='relu', padding='same')(x)
    x = Conv1D(num_cp * 128, 3, activation='relu', padding='same')(x)
    x = Conv1D(num_cp * 128, 3, activation='relu', padding='same')(x)
    latent_space = x

    concat = monaco_plan(latent_space, num_cp)

    return Model(inp, concat)
    
    

def monaco_plan(latent_space, num_cp):
    leafs = []
    leaf_total = tf.split(Reshape((num_cp, 2, 64))(latent_space), num_cp, axis=1)
    for i in range(num_cp):
        leafs.append(Activation('sigmoid', name=f'leaf_{i}')(leaf_total[i]))
    
    mus = []
    mu_total = Conv1D(num_cp * 32, 3, activation='relu', padding='same')(latent_space)
    mu_total = Conv1D(num_cp * 8, 3, activation='relu', padding='same')(mu_total)
    mu_total = Conv1D(num_cp, 1, activation='relu', padding='same')(mu_total)
    mu_total = tf.split(Reshape((num_cp, -1))(mu_total), num_cp, axis=1)
    for i in range(num_cp):
        mus.append(Activation('sigmoid', name=f'mu_{i}')(mu_total[i]))
            
    return monaco_param_to_vector(leafs, mus)