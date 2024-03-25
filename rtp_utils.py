import numpy as np
import matplotlib.pyplot as plt
from rtp_data import generate_data
from rtp_model import *
from matplotlib import animation
import matplotlib
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.compat.v1.enable_eager_execution()

 
class save_gif():
    def __init__(self, model, seg_model, decoder_info, structure_weights, cp_idx, num_cp, experiment, epoch):
        matplotlib.use('agg')
        save_path = f"imgs/{cp_idx}.gif"
        x, y = generate_data(seg_model, 1, 0)
        _, num_cp, dose_shape, img_shape, num_slices, leaf_length = decoder_info
        decoder = MonacoDecoder(num_cp, dose_shape, img_shape, num_slices, leaf_length)
        monaco_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("mlc").output, model.get_layer(f"ray_{cp_idx}").output, model.get_layer(f"mus").output, model.layers[-1].output])

        mlcs, _, mus, dose = monaco_model.predict_on_batch(x)     
        absorption_matrices = decoder.get_absorption_matrices(x[..., 0:1])   
        ray_strength = decoder.get_rays(absorption_matrices, mlcs, mus)
        self.delivered_dose = dose[0, ..., 0]
        self.ground_truth = y[0, ..., 0]
        self.ct = x[0, ..., 0]

        self.structure_weights = structure_weights
        self.weights = np.ones_like(self.ct)
        for i in range(len(structure_weights)):
            self.weights *= structure_weights[i]

        # self.mus = mus
        self.experiment = experiment
        self.epoch = epoch
        self.save_path = save_path
        self.mlc = mlcs[0, ..., cp_idx]
        self.ray_matrix = ray_strength[cp_idx].numpy()[0, ...]

        # print("Lower leafs: ", leafs[0, 0, :])
        # print("Upper leafs: ", leafs[0, 1, :])
        # self.ray_matrix = np.ones_like(ray_matrix)
        # for i in range(ray_matrix.shape[2]):
        #     self.ray_matrix[:, :, i] = tf.gather(self.mlc[i, :], tf.cast(ray_matrix[..., i], dtype=tf.int32))

        self.fig, axx = plt.subplots(2, 2)
        self.fps = 12
        self.num_frames = self.ct.shape[2]
        self.makegif(save_path, experiment, epoch)
    
    def makegif(self, save_path, experiment, epoch):
        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                        frames=self.num_frames)
        anim.save(save_path, writer=animation.PillowWriter(fps=self.fps, codec='libx264'))
        experiment.log_image(save_path, step=epoch, overwrite=True)
        plt.close()
    
    def animate(self, i):
        dose_slice = self.delivered_dose[:, :, i]
        gt_slice = self.ground_truth[:, :, i]
        ray_slice = self.ray_matrix[:, :, i]
        ct_slice = self.ct[:, :, i]
        weight_slice = self.weights[:, :, i]

        self.im2.set_array(ray_slice)
        self.im3.set_array(gt_slice)
        self.im4.set_array(dose_slice)
        self.im3ct.set_array(ct_slice)
        self.im4ct.set_array(ct_slice)
        self.imw.set_array(weight_slice)
        return [self.im2, self.im3, self.im4, self.im3ct, self.im4ct, self.imw]

    def init(self):
        mono_font = {'fontname':'monospace'}
        self.title = self.fig.suptitle("", fontsize=4, **mono_font)
        vmin = np.min(self.ground_truth)
        vmax = np.max(self.ground_truth)
        dose_slice = self.delivered_dose[:, :, 0]
        gt_slice = self.ground_truth[:, :, 0]
        ray_slice = self.ray_matrix[:, :, 0]
        ct_slice = self.ct[:, :, 0]
        weight_slice = self.weights[:, :, 0]
        
        plt.subplot(221)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        plt.imshow(self.mlc, cmap="gray", vmin=0, vmax=1, interpolation="none")


        plt.subplot(222)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im2 = plt.imshow(ray_slice, cmap="gray", vmin=0, vmax=np.max(self.ray_matrix), interpolation="none")


        plt.subplot(234)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.imw = plt.imshow(weight_slice, vmin=1, vmax=np.max(self.structure_weights), cmap="hot", interpolation='bilinear')

        plt.subplot(235)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im3ct = plt.imshow(ct_slice, vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
        self.im3 = plt.imshow(gt_slice, alpha=.5, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")


        plt.subplot(236)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im4ct = plt.imshow(ct_slice, vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
        self.im4 = plt.imshow(dose_slice, alpha=.5, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")


    

def get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point):
    dose = 0.0
    for cp_idx in range(len(ray_matrices)):
        absorption_value = absorption_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2], 0]
        ray_idx = tf.cast(ray_matrices[cp_idx][0, mc_point[0], mc_point[1], mc_point[2]], dtype=tf.int32)
        leaf_cond = leafs[mc_point[2], ray_idx, cp_idx]
        dep_value = absorption_value * mus[0, cp_idx] * leaf_cond
        dose += dep_value
    return dose

