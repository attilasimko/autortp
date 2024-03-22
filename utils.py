import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from data import generate_data
from matplotlib import animation
import matplotlib
import tensorflow as tf
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.compat.v1.enable_eager_execution()

 
class save_gif():
    def __init__(self, absorption_matrix, delivered_dose, ground_truth, ray_strengths, leafs, experiment, epoch, save_path):
        matplotlib.use('agg')
        self.absorption_matrix = absorption_matrix[..., 0]
        self.delivered_dose = delivered_dose
        self.ground_truth = ground_truth[0, ..., 0]
        self.leafs = leafs
        # self.mus = mus
        self.experiment = experiment
        self.epoch = epoch
        self.save_path = save_path
        self.mlc = self.leafs[0, ...] # tf.where(tf.greater(self.leafs[0, ...], 0.5), 1.0, 0.0)
        self.ray_matrix = ray_strengths

        # print("Lower leafs: ", leafs[0, 0, :])
        # print("Upper leafs: ", leafs[0, 1, :])
        # self.ray_matrix = np.ones_like(ray_matrix)
        # for i in range(ray_matrix.shape[2]):
        #     self.ray_matrix[:, :, i] = tf.gather(self.mlc[i, :], tf.cast(ray_matrix[..., i], dtype=tf.int32))

        self.fig, axx = plt.subplots(2, 2)
        self.fps = 12
        self.num_frames = ground_truth.shape[3]
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

        self.im2.set_array(ray_slice)
        self.im3.set_array(gt_slice)
        self.im4.set_array(dose_slice)
        return [self.im2, self.im3, self.im4]

    def init(self):
        mono_font = {'fontname':'monospace'}
        self.title = self.fig.suptitle("", fontsize=4, **mono_font)
        vmin = np.min(self.ground_truth)
        vmax = np.max(self.ground_truth)
        dose_slice = self.delivered_dose[:, :, 0]
        gt_slice = self.ground_truth[:, :, 0]
        ray_slice = self.ray_matrix[:, :, 0]
        
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
        self.im2 = plt.imshow(ray_slice, cmap="gray", vmin=0, vmax=1, interpolation="none")


        plt.subplot(223)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im3 = plt.imshow(gt_slice, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")
        plt.colorbar()


        plt.subplot(224)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im4 = plt.imshow(dose_slice, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")
        plt.colorbar()

def get_rays(ray_matrices, absorption_matrices, leafs):
    ray_strengths = []
    for batch_idx in range(leafs.shape[0]):
        for cp_idx in range(len(ray_matrices)):
            ray_slices = []
            for slice_idx in range(leafs.shape[2]):
                current_leafs = leafs[batch_idx, ..., slice_idx, cp_idx]
                ray_matrix = tf.cast(ray_matrices[cp_idx][batch_idx, ...], dtype=tf.int32)
                ray_slices.append(tf.gather(current_leafs, ray_matrix))
            ray_stack = tf.stack(ray_slices, -1)
            absorbed_rays = tf.multiply(ray_stack, absorption_matrices[cp_idx][batch_idx, :, :, 0])
            ray_strengths.append(absorbed_rays)
    return ray_strengths

def plot_res(experiment, model, ray_matrices, leaf_length, num_cp, epoch):
    x, y = generate_data(1, (32, 32, 32), 20)
    num_step = 4
    dose = np.zeros_like(y)[0, ..., 0]

    pred = model.predict_on_batch(x)

    absorption_matrices = np.split(get_absorption_matrices(x[..., 0:1], num_cp), num_cp, -1)
    leafs = vector_to_monaco_param(pred, leaf_length, num_cp)
    ray_strengths = get_rays(ray_matrices, absorption_matrices, leafs)
    dose = tf.reduce_sum(ray_strengths, axis=0)

    for i in range(num_cp):
        save_gif(absorption_matrices[i][0, ...], dose, y, ray_strengths[i], leafs[..., i], experiment, epoch, f"imgs/{i}.gif")
    

def get_absorption_matrices(ct, num_cp):
    batches = []
    absorption_scalar = 1 / (96)
    absorption = np.ones_like(ct) * absorption_scalar
    for batch in range(ct.shape[0]):
        rotated_arrays = []
        for idx in range(num_cp):
            array = rotate(absorption[batch, ...], idx * 360 / num_cp, (0, 1), reshape=False, order=0, mode='nearest')
            array = 1 - tf.cumsum(array, axis=0)
            array = rotate(array, - idx * 360 / num_cp, (0, 1), reshape=False, order=0, mode='nearest')
            array = tf.where(tf.greater(array, 0), array, 0)
            rotated_arrays.append(array)
        batches.append(tf.stop_gradient(tf.constant(tf.cast(tf.concat(rotated_arrays, axis=-1), dtype=tf.float32))))

    return tf.stack(batches, 0)

def get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point):
    dose = 0.0
    for cp_idx in range(len(ray_matrices)):
        absorption_value = absorption_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2], 0]
        ray_idx = tf.cast(ray_matrices[cp_idx][0, mc_point[0], mc_point[1], mc_point[2]], dtype=tf.int32)
        leaf_cond = leafs[mc_point[2], ray_idx, cp_idx]
        dep_value = absorption_value * mus[0, cp_idx] * leaf_cond
        dose += dep_value
    return dose

def get_monaco_projections(num_cp):
    shape = (64, 64)
    rotated_arrays = []
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indeces = x

    for angle_idx in range(num_cp):
        array = np.expand_dims(rotate(indeces, - angle_idx * 360 / num_cp, (0, 1), reshape=False, order=0, mode='nearest'), 0)
        rotated_arrays.append(array)

    return [tf.constant(x) for x in rotated_arrays]

def monaco_param_to_vector(leafs): #, mus):
    # mus = tf.stack(mus, -1)

    return tf.reshape(leafs, [leafs.shape[0], -1]) # tf.concat([tf.reshape(leafs, [-1]), tf.reshape(mus, [-1])], -1)

def vector_to_monaco_param(vec, leaf_length=64, num_cp=6, batch_size=1):
    leafs = tf.reshape(vec[:batch_size * leaf_length * 64 * num_cp], [batch_size, 64, leaf_length, num_cp])
    # mus = tf.reshape(vec[batch_size * leaf_length * 64 * num_cp:], [batch_size, 1, num_cp])
    return leafs # , mus