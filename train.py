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
# tf.compat.v1.enable_eager_execution()
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ''


n_epochs = 100
epoch_length = 1000
batch_size = 1

# Number of control points
num_cp = 6
# Length of each leaf
leaf_length = 64
# Number of Monte Carlo points
num_mc = 1000
# Generate ray matrices for each control point
ray_matrices = get_monaco_projections(num_cp)

model = build_model(batch_size, num_cp)
model.compile(loss=rtp_loss(num_cp, num_mc, leaf_length), optimizer=Adam(learning_rate=0.000001))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

for epoch in range(n_epochs):
    training_loss = []
    for i in range(epoch_length):
        x, y = generate_data()
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    plot_res(model, ray_matrices, leaf_length, num_cp, epoch)