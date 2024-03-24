import numpy as np
import matplotlib.pyplot as plt
from data import generate_data
from matplotlib import animation
import matplotlib
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.compat.v1.enable_eager_execution()

 
class save_gif():
    def __init__(self, ct, mlc, delivered_dose, ground_truth, ray_strengths, experiment, epoch, save_path):
        matplotlib.use('agg')
        self.delivered_dose = delivered_dose
        self.ground_truth = ground_truth
        self.ct = ct
        # self.mus = mus
        self.experiment = experiment
        self.epoch = epoch
        self.save_path = save_path
        self.mlc = mlc
        self.ray_matrix = ray_strengths

        # print("Lower leafs: ", leafs[0, 0, :])
        # print("Upper leafs: ", leafs[0, 1, :])
        # self.ray_matrix = np.ones_like(ray_matrix)
        # for i in range(ray_matrix.shape[2]):
        #     self.ray_matrix[:, :, i] = tf.gather(self.mlc[i, :], tf.cast(ray_matrix[..., i], dtype=tf.int32))

        self.fig, axx = plt.subplots(2, 2)
        self.fps = 12
        self.num_frames = ground_truth.shape[2]
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

        self.im2.set_array(ray_slice)
        self.im3.set_array(gt_slice)
        self.im4.set_array(dose_slice)
        self.im3ct.set_array(ct_slice)
        self.im4ct.set_array(ct_slice)
        return [self.im2, self.im3, self.im4, self.im3ct, self.im4ct]

    def init(self):
        mono_font = {'fontname':'monospace'}
        self.title = self.fig.suptitle("", fontsize=4, **mono_font)
        vmin = np.min(self.ground_truth)
        vmax = np.max(self.ground_truth)
        dose_slice = self.delivered_dose[:, :, 0]
        gt_slice = self.ground_truth[:, :, 0]
        ray_slice = self.ray_matrix[:, :, 0]
        ct_slice = self.ct[:, :, 0]
        
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


        plt.subplot(223)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im3ct = plt.imshow(ct_slice, vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
        self.im3 = plt.imshow(gt_slice, alpha=.5, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")


        plt.subplot(224)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im4ct = plt.imshow(ct_slice, vmin=-1, vmax=1, cmap="gray", interpolation='bilinear')
        self.im4 = plt.imshow(dose_slice, alpha=.5, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")


def plot_res(experiment, model, num_cp, epoch):
    x, y = generate_data(1, 0)
    for i in range(num_cp):
        monaco_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("mlc").output, model.get_layer(f"ray_{i}").output, model.layers[-1].output])
        mlcs, ray_strength, dose = monaco_model.predict_on_batch(x)
        # pred = tf.where(tf.greater(pred, 0.5), 1.0, 0.0)
        

        save_gif(x[0, :, :, :, 0], mlcs[0, ..., i], dose[0, ...], y[0, ..., 0], ray_strength[0, ...], experiment, epoch, f"imgs/{i}.gif")
    

def get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point):
    dose = 0.0
    for cp_idx in range(len(ray_matrices)):
        absorption_value = absorption_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2], 0]
        ray_idx = tf.cast(ray_matrices[cp_idx][0, mc_point[0], mc_point[1], mc_point[2]], dtype=tf.int32)
        leaf_cond = leafs[mc_point[2], ray_idx, cp_idx]
        dep_value = absorption_value * mus[0, cp_idx] * leaf_cond
        dose += dep_value
    return dose

