import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Reshape
tf.keras.mixed_precision.set_global_policy('mixed_float16')
import numpy as np
from utils import get_monaco_projections, monaco_param_to_vector


def build_model(batch_size=1, num_cp=6):
    inp = Input(shape=(64, 64, 64, 2), dtype=tf.float16, batch_size=batch_size)
    x = Conv3D(4, (3, 3, 3), activation='relu', padding='same')(inp)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((4, 4, 4))(x)
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    latent_space = Flatten()(x)


    concat = monaco_plan(inp[..., 0], latent_space, num_cp)

    return Model(inp, concat)
    
    

def monaco_plan(ct, latent_space, num_cp):
    leafs = []
    for i in range(num_cp):
        leaf = Dense(128, activation='sigmoid')(latent_space)
        leafs.append(Reshape((2, 64), name=f'leaf_{i}')(leaf))
    
    mus = []
    for i in range(num_cp):
        mu = Dense(1, activation='relu')(latent_space)
        mus.append(mu)
            
    return monaco_param_to_vector(leafs, mus)