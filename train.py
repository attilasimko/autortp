from data import generate_data
from model import build_model
from losses import rtp_loss
import matplotlib.pyplot as plt
from utils import *
import numpy as np
import os 
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

def plot_res(ray_matrices, num_cp):
    x, y = generate_data((32, 32, 32), 10)
    dose = np.zeros_like(y)
    pred = model.predict_on_batch(x)
    # pred = np.random.rand(774)
    absorption_matrices = get_absorption_matrices(y, num_cp)
    leafs, mus = vector_to_monaco_param(pred)
    # num_step = 4
    # for i in range(0, 64, num_step):
    #     for j in range(0, 64, num_step):
    #         for k in range(0, 64, num_step):
    #             mc_point = tf.constant([i, j, k], dtype=tf.int32)
    #             dose[0, i:i+num_step, j:j+num_step, k:k+num_step] = get_dose_value(num_cp, absorption_matrices, ray_matrices, leafs, mus, mc_point)

    num_slices = 6
    for i in range(num_slices):
        plt.subplot(4, 8, i * 2 + 1)
        plt.title(f"{int(i * 64 / num_slices)}-gt")
        plt.imshow(ray_matrices[i][0, :, :, 0])#int(i * 64 / num_slices)])#, vmin=0, vmax=5)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 8, i * 2 + 2)
        plt.title(f"{int(i * 64 / num_slices)}-pred")
        plt.imshow(ray_matrices[i][0, :, :, 63])#int(i * 64 / num_slices)])#, vmin=0, vmax=5)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'imgs/{epoch}.png')

n_epochs = 10
epoch_length = 10000

num_cp = 6
num_mc = 100
ray_matrices = get_monaco_projections(num_cp)

model = build_model()
model.compile(loss=rtp_loss(num_cp, num_mc), optimizer=Adam(learning_rate=0.0001))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

for epoch in range(n_epochs):
    training_loss = []
    # for i in range(epoch_length):
    #     x, y = generate_data()
    #     loss = model.train_on_batch(x, y)
    #     training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    plot_res(ray_matrices, num_cp)