from comet_ml import Experiment
import sys
import os
import os
import argparse
import numpy as np
from rtp_data import DataGenerator
from rtp_model import build_model
from rtp_losses import rtp_loss
from rtp_utils import *
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Model
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='Welcome.')
parser.add_argument("--base", default=None)
args = parser.parse_args()

if (args.base == "GAUSS"):
    base_path = '/data/attila/ARTP/'
else:
    base_path = '/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/DSets/ARTP/'

n_epochs = 500
epoch_length = 50
batch_size = 1
learning_rate = 0.0001
structs = ["Bladder", "FemoralHead_L", "FemoralHead_R", "Rectum"]
weights = [3.0, 2.0, 2.0, 2.0, 2.0]

img_size = (128, 128, 240)

# Custom data generator for efficiently loading the training data (stored as .npz files under base_path+"training/")
gen_train = DataGenerator(base_path + "training/",
                          structs=structs,
                          img_size=img_size,
                          batch_size=batch_size)

# Data generator for validation data
gen_val = DataGenerator(base_path + "validating/",
                        structs=structs,
                          img_size=img_size,
                        batch_size=batch_size)

# Data generator for test data
gen_test = DataGenerator(base_path + "testing/",
                         structs=structs,
                          img_size=img_size,
                         batch_size=batch_size)
    
# Comet_ml
experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="gerd")

# (Decoder type / Number of control points / Dose resolution / Image shape / Ray length / Leaf resolution)
decoders = [("umea", 24, 64, img_size, 60, 128), ("vienna", 24, 64, img_size, 80, 128)]
models = build_model(batch_size, img_size, decoders)
for i in range(len(decoders)):
    models[i].compile(loss=rtp_loss(weights), optimizer=Adam(learning_rate=learning_rate))
    print(f"{decoders[i][0]} - Model parameters: {int(np.sum([K.count_params(p) for p in models[i].trainable_weights]))}")

# Debug part
# x, y = gen_train[0]
# monaco_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("mlc").output, model.get_layer(f"ray_{0}").output, model.layers[-1].output])
# x = tf.Variable(x, dtype=tf.float32)
# with tf.GradientTape(persistent=True) as tape:
#     predictions = model(x)
#     loss_value = model.loss(tf.Variable(y, dtype=tf.float32), predictions)   
# gradients = tape.gradient(loss_value, predictions)

for epoch in range(n_epochs):
    training_loss = []
    val_loss = []
    for i in range(len(decoders)):
        training_loss.append([])
        val_loss.append([])

    for _ in range(epoch_length):
        for idx in range(len(gen_train)):
            x, y = gen_train[idx]
            for i in range(len(decoders)):
                loss = models[i].train_on_batch(x, y)
                training_loss[i].append(loss)

    print(f'Epoch {epoch + 1}/{n_epochs} - training loss: {[np.mean(curr_loss) for curr_loss in training_loss]}')
    experiment.log_metric("train_loss", np.mean(training_loss), step=epoch)

    for idx in range(len(gen_val)):
        x, y = gen_val[idx]
        for i in range(len(decoders)):
            loss = models[i].test_on_batch(x, y)
            val_loss[i].append(loss)

    print(f'\t\tvalidation loss: {[np.mean(curr_loss) for curr_loss in val_loss]}')
    for i in range(len(decoders)):
        experiment.log_metric(f"val_loss_{decoders[i][0]}", np.mean(val_loss[i]), step=epoch)

    save_gif(gen_val, models, decoders, weights, experiment, epoch)