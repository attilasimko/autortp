from data import generate_data
from model import build_model
from losses import rtp_loss
import matplotlib.pyplot as plt
import numpy as np
import os 
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def plot_res():
    x, y = generate_data((32, 32), 10)
    pred = model.predict_on_batch(x)
    num_slices = 16
    for i in range(num_slices):
        plt.subplot(4, 8, i * 2 + 1)
        plt.title(f"{int(i * 64 / num_slices)}-gt")
        plt.imshow(y[0, :, :, int(i * 64 / num_slices)], vmin=0, vmax=5)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 8, i * 2 + 2)
        plt.title(f"{int(i * 64 / num_slices)}-pred")
        plt.imshow(pred[0, :, :, int(i * 64 / num_slices)], vmin=0, vmax=5)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'imgs/{epoch}.png')

n_epochs = 10
epoch_length = 1000

model = build_model()
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

for epoch in range(n_epochs):
    training_loss = []
    for i in range(epoch_length):
        x, y = generate_data()
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    plot_res()