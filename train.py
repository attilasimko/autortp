from data import generate_data
from model import build_model
from losses import rtp_loss
from utils import *
import numpy as np
import os 
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

n_epochs = 5
epoch_length = 1000
batch_size = 6
learning_rate = 0.00001

# Number of control points
num_cp = 12
# Length of each leaf
leaf_length = 64
# Number of Monte Carlo points
num_mc = 50
# Generate ray matrices for each control point
ray_matrices = get_monaco_projections(num_cp)

model = build_model(batch_size, num_cp)
model.compile(loss=rtp_loss(ray_matrices, num_cp, num_mc, leaf_length), optimizer=Adam(learning_rate=learning_rate))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

for epoch in range(n_epochs):
    training_loss = []
    for i in range(epoch_length):
        x, y = generate_data(batch_size)
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    plot_res(model, ray_matrices, leaf_length, num_cp, epoch)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')