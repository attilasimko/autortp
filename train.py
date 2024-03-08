from comet_ml import Experiment
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

n_epochs = 50
epoch_length = 5
batch_size = 6
learning_rate = 0.00001

# Number of control points
num_cp = 12         
# Length of each leaf
leaf_length = 64
# Number of Monte Carlo points
ratio_mc = 0.01
num_mc = int(ratio_mc * (64*64*64))
# Generate ray matrices for each control point
ray_matrices = get_monaco_projections(num_cp)

# Comet_ml
experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="gerd")

# Debug part
# loss = rtp_loss(ray_matrices, num_cp, num_mc, leaf_length)
# x, y = generate_data(batch_size)
# y = np.concatenate([y, get_absorption_matrices(x[..., 0:1], num_cp)], -1)
# pred = np.random.rand(batch_size * leaf_length * 2 * num_cp + batch_size * num_cp, dtype=np.float16)
# loss(y, pred)

model = build_model(batch_size, num_cp)
model.compile(loss=rtp_loss(ray_matrices, num_cp, num_mc, leaf_length), optimizer=Adam(learning_rate=learning_rate))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

# Debug part

for epoch in range(n_epochs):
    training_loss = []
    for i in range(epoch_length):
        x, y = generate_data(batch_size)
        y = np.concatenate([y, get_absorption_matrices(x[..., 0:1], num_cp)], -1)
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    experiment.log_metric("loss", np.mean(training_loss), step=epoch)
    plot_res(experiment, model, ray_matrices, leaf_length, num_cp, epoch)