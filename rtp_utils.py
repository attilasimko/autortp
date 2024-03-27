import numpy as np
import matplotlib.pyplot as plt
from rtp_model import *
from matplotlib import animation
import matplotlib
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.compat.v1.enable_eager_execution()

 
class save_gif():
    def __init__(self, gen, models, decoders, structure_weights, experiment, epoch):
        matplotlib.use('agg')
        self.experiment = experiment
        self.epoch = epoch
        self.save_path = f"imgs/results.gif"
        self.num_cols = len(models) + 2
        self.fig, axx = plt.subplots(1, self.num_cols, figsize=(self.num_cols*4, 4))
        self.fps = 12

        x, y = gen[0]

        self.models = models
        self.decoders = decoders
        self.ground_truth = y[0, ..., 0]
        self.ct = x[0, ..., 0]
        self.structure_weights = structure_weights
        self.weights = np.ones_like(self.ct)
        for i in range(len(structure_weights)):
            self.weights = np.where(y[0, ..., i+1] > 0.5, structure_weights[i], self.weights)
        self.num_frames = self.ct.shape[2]

        self.delivered_dose = []
        for i in range(len(models)):
            model = models[i]
            decoder_info = decoders[i]

            name, num_cp, dose_shape, img_shape, num_slices, leaf_length = decoder_info
            decoder = MonacoDecoder(num_cp, dose_shape, img_shape, num_slices, leaf_length)
            monaco_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("mlc").output, model.get_layer(f"mus").output, model.layers[-1].output])

            mlcs, mus, dose = monaco_model.predict_on_batch(x)     
            mlcs = tf.where(mlcs > 0.2, 1.0, 0.0)
            self.delivered_dose.append(dose[0, ..., 0])
        self.delivered_dose = np.stack(self.delivered_dose, -1)
            

        self.makegif(experiment, epoch)
    
    def makegif(self, experiment, epoch):
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                        frames=self.num_frames)
        anim.save(self.save_path, writer=animation.PillowWriter(fps=self.fps, codec='libx264'))
        experiment.log_image(self.save_path, step=epoch, overwrite=True)
        plt.close()
    
    def animate(self, i):
        dose_slice = self.delivered_dose[:, :, i, :]
        gt_slice = self.ground_truth[:, :, i]
        ct_slice = self.ct[:, :, i]
        weight_slice = self.weights[:, :, i]

        self.im3.set_array(gt_slice)
        for i in range(len(self.models)):
            self.im4[i].set_array(dose_slice[..., i])
            self.im4ct[i].set_array(ct_slice)
        self.im3ct.set_array(ct_slice)
        self.imw.set_array(weight_slice)
        return [self.im3, self.im4, self.im3ct, self.im4ct, self.imw]

    def init(self):
        mono_font = {'fontname':'monospace'}
        self.title = self.fig.suptitle("", fontsize=4, **mono_font)
        vmin = np.min(self.ground_truth)
        vmax = np.max(self.ground_truth)
        dose_slice = self.delivered_dose[:, :, 0, :]
        gt_slice = self.ground_truth[:, :, 0]
        ct_slice = self.ct[:, :, 0]
        weight_slice = self.weights[:, :, 0]
        
        plt.subplot(1, self.num_cols, 1)
        plt.title('weights')
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.imw = plt.imshow(weight_slice, vmin=1, vmax=np.max(self.structure_weights), cmap="hot", interpolation='bilinear')

        plt.subplot(1, self.num_cols, 2)
        plt.title('true_dose')
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im3ct = plt.imshow(ct_slice, vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
        self.im3 = plt.imshow(gt_slice, alpha=.5, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")

        self.im4 = []
        self.im4ct = []
        for i in range(len(self.models)):
            plt.subplot(1, self.num_cols, 3 + i)
            plt.title(f'{self.decoders[i][0]}')
            plt.tick_params(left=False,
                            bottom=False,
                            labelleft=False,
                            labelbottom=False)
            self.im4ct.append(plt.imshow(ct_slice, vmin=-1, vmax=1, cmap="gray", interpolation='bilinear'))
            self.im4.append(plt.imshow(dose_slice[..., i], alpha=.5, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none"))


    

def get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point):
    dose = 0.0
    for cp_idx in range(len(ray_matrices)):
        absorption_value = absorption_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2], 0]
        ray_idx = tf.cast(ray_matrices[cp_idx][0, mc_point[0], mc_point[1], mc_point[2]], dtype=tf.int32)
        leaf_cond = leafs[mc_point[2], ray_idx, cp_idx]
        dep_value = absorption_value * mus[0, cp_idx] * leaf_cond
        dose += dep_value
    return dose

