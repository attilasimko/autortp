from comet_ml import Experiment
from data import generate_data
from model import build_model
from utils import *
import numpy as np
import os 
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.losses import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Model
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

n_epochs = 50
epoch_length = 10000
batch_size = 1
learning_rate = 0.0001

# Number of control points
num_cp = 24

# Comet_ml
experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="gerd")

x, y = generate_data(batch_size, 0)
model = build_model(batch_size, x.shape, num_cp, "monaco")
model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=learning_rate))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

# Debug part
# x, y = generate_data(batch_size, 0)
# monaco_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("mlc").output, model.get_layer(f"ray_{0}").output, model.layers[-1].output])
# monaco_model.compile(loss=mean_absolute_error, optimizer=Adam(learning_rate=learning_rate))
# x = tf.Variable(x, dtype=tf.float32)
# with tf.GradientTape(persistent=True) as tape:
#     predictions = monaco_model(x)[0]
#     loss_value = monaco_model.loss(tf.Variable(y, dtype=tf.float32), predictions)   
# gradients = tape.gradient(loss_value, predictions)

for epoch in range(n_epochs):
    training_loss = []
    for i in range(epoch_length):
        x, y = generate_data(batch_size)
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    experiment.log_metric("loss", np.mean(training_loss), step=epoch)
    for i in range(num_cp):
        save_gif(model, i, num_cp, experiment, epoch)