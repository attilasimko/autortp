from comet_ml import Experiment
import sys
import os
from rtp_data import generate_data
from rtp_model import build_model
from rtp_losses import rtp_loss
from rtp_utils import *
sys.path.append("../miqa-seg")
import os
import numpy as np
from utils import model_config
from model import build_unet
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Model
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

n_epochs = 50
epoch_length = 100
batch_size = 1
learning_rate = 0.0001

# Number of control points
num_cp = 24

# Comet_ml
experiment = Experiment(api_key="ro9UfCMFS2O73enclmXbXfJJj", project_name="gerd")

num_filters = 48
config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
seg_path = "weights/crimson_cliff_1296_hero.h5"
seg_model = build_unet(len(config.labels), 1, num_filters, 0.0, config.epsilon)
seg_model.load_weights(seg_path, skip_mismatch=True, by_name=True)
seg_model = Model(inputs=seg_model.input, outputs=[tf.expand_dims(seg_model.layers[-1].output[..., config.maps["Bladder"]], -1),
                                                   tf.expand_dims(seg_model.layers[-1].output[..., config.maps["FemoralHead_L"]], -1),
                                                   tf.expand_dims(seg_model.layers[-1].output[..., config.maps["FemoralHead_R"]], -1),
                                                   tf.expand_dims(seg_model.layers[-1].output[..., config.maps["Rectum"]], -1)])
seg_model.compile(loss="mse")
seg_model.trainable = False
for l in seg_model.layers:
    l.trainable = False

weights = [10.0, 3.0, 8.0, 8.0, 3.0]
x, y = generate_data(seg_model, batch_size, 0)
# (Decoder type / Number of control points / Dose shape / Image shape / Number of slices / Leaf length)
decoder_info = ("monaco", num_cp, (64, 64), (256, 256), x.shape[3], 32)
model = build_model(batch_size, x.shape, decoder_info)
model.compile(loss=rtp_loss(weights), optimizer=Adam(learning_rate=learning_rate))
print(f"Number of model parameters: {int(np.sum([K.count_params(p) for p in model.trainable_weights]))}")

# Debug part
# x, y = generate_data(seg_model, batch_size, 0)
# # monaco_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("mlc").output, model.get_layer(f"ray_{0}").output, model.layers[-1].output])
# x = tf.Variable(x, dtype=tf.float32)
# with tf.GradientTape(persistent=True) as tape:
#     predictions = model(x)
#     loss_value = model.loss(tf.Variable(y, dtype=tf.float32), predictions)   
# gradients = tape.gradient(loss_value, predictions)

for epoch in range(n_epochs):
    training_loss = []
    x, y = generate_data(seg_model, batch_size)
    for i in range(epoch_length):
        loss = model.train_on_batch(x, y)
        training_loss.append(loss)
    print(f'Epoch {epoch + 1}/{n_epochs} - loss: {np.mean(training_loss)}')
    experiment.log_metric("loss", np.mean(training_loss), step=epoch)
    for i in range(num_cp):
        save_gif(model, seg_model, decoder_info, weights, i, num_cp, experiment, epoch)