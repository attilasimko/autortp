from data import generate_data
from model import build_model
from losses import rtp_loss
import matplotlib.pyplot as plt
import numpy as np
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def plot_res(x, y):
    pred = model.predict(x)
    num_slices = 16
    for i in range(num_slices):
        plt.subplot(4, 8, i * 2 + 1)
        plt.imshow(y[0, :, :, int(i * 64 / num_slices)], vmin=0, vmax=5)
        plt.subplot(4, 8, i * 2 + 2)
        plt.imshow(pred[0, :, :, int(i * 64 / num_slices)], vmin=0, vmax=5)
    plt.savefig(f'imgs/{epoch}.png')

n_epochs = 10
epoch_length = 10000

model = build_model()
model.compile(loss='mae', optimizer='Adam')
for epoch in range(n_epochs):
    training_loss = []
    for i in range(epoch_length):
        x, y = generate_data()
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    plot_res(generate_data()[0])