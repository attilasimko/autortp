import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from data import generate_data
from matplotlib import animation
import matplotlib
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

 
class save_gif():
    def __init__(self, absorption_matrix, delivered_dose, ground_truth, ray_matrix, leafs, mus, experiment, epoch, save_path):
        matplotlib.use('agg')
        self.absorption_matrix = absorption_matrix[..., 0]
        self.delivered_dose = delivered_dose
        self.ground_truth = ground_truth[0, ..., 0]
        self.leafs = leafs
        self.mus = mus
        self.experiment = experiment
        self.epoch = epoch
        self.save_path = save_path
        self.mlc = np.zeros((64, 64), dtype=np.bool)
        for i in range(64):
            for j in range(64):
                if i > leafs[0, 0, j] * 32 and i < 64 - leafs[0, 1, j] * 32:
                    self.mlc[i, j] = True

        # print("Lower leafs: ", leafs[0, 0, :])
        # print("Upper leafs: ", leafs[0, 1, :])
        self.ray_matrix = np.zeros_like(ray_matrix)
        for i in range(ray_matrix.shape[2]):
            self.ray_matrix[:, :, i] = np.isin(ray_matrix[:, :, i], range(int((i * 64) + (leafs[0, 0, i] * 32)), int((i * 64) + (64 - leafs[0, 1, i] * 32)))).astype(np.float16)
        self.ray_matrix = np.where(self.ray_matrix > 0, self.mus[0, 0], 0)

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


        plt.subplot(224)
        plt.tick_params(left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False)
        self.im4 = plt.imshow(dose_slice, cmap="jet", vmin=vmin, vmax=vmax, interpolation="none")


def plot_res(experiment, model, ray_matrices, leaf_length, num_cp, epoch):
    x, y = generate_data(1, (32, 32, 32), 20)
    num_step = 2
    dose = np.zeros_like(y)[0, ..., 0]

    # pred = np.array(np.random.rand(leaf_length * 2 * num_cp + num_cp), dtype=np.float16)
    pred = model.predict_on_batch(x)

    # pred = np.zeros((774))
    absorption_matrices = np.split(get_absorption_matrices(x[..., 0:1], num_cp), num_cp, -1)
    leafs, mus = vector_to_monaco_param(pred, leaf_length, num_cp)
    
    for i in range(0, 64, num_step):
        for j in range(0, 64, num_step):
            for k in range(0, 64, num_step):
                mc_point = tf.constant([i, j, k], dtype=tf.int32)
                dose[i:i+num_step, j:j+num_step, k:k+num_step] = get_dose_value([matrix[0, ...] for matrix in absorption_matrices], ray_matrices, leafs[0, ...], mus[0, ...], mc_point)
    
    for i in range(num_cp):
        save_gif(absorption_matrices[i][0, ...], dose, y, ray_matrices[i][0, ...], leafs[..., i], mus[..., i], experiment, epoch, f"imgs/{i}.gif")
    

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
            rotated_arrays.append(tf.cast(array, dtype=np.float16))
        batches.append(tf.concat(rotated_arrays, axis=-1))

    return tf.stack(batches, 0)

def get_dose_value(absorption_matrices, ray_matrices, leafs, mus, mc_point):
    dose = 0.0
    for cp_idx in range(len(ray_matrices)):
        absorption_value = absorption_matrices[cp_idx][mc_point[0], mc_point[1], mc_point[2], 0]
        leaf_idx = mc_point[2]
        ray_idx = tf.cast(ray_matrices[cp_idx][0, mc_point[0], mc_point[1], mc_point[2]], dtype=tf.float16)
        ray_idx -= tf.cast(64 * leaf_idx, dtype=tf.float16)
        cond_leafs = tf.logical_and(tf.greater_equal(ray_idx, leafs[0, leaf_idx, cp_idx] * 32), tf.less_equal(ray_idx, 64 - leafs[1, leaf_idx, cp_idx] * 32))
        dep_value = mus[0, cp_idx] * absorption_value
        dose += tf.reduce_sum(tf.where(cond_leafs, dep_value, 0))
    return dose

def get_monaco_projections(num_cp):
    shape = (64, 64)
    rotated_arrays = []
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indeces = y * shape[0] + x
    indeces = np.repeat(indeces[..., None], 64, -1)
    indeces = np.swapaxes(indeces, 0, 2)

    for angle_idx in range(num_cp):
        array = np.expand_dims(rotate(indeces, angle_idx * 360 / num_cp, (0, 1), reshape=False, order=0, mode='nearest'), 0)
        rotated_arrays.append(array)

    return [tf.constant(x, dtype=tf.float16) for x in rotated_arrays]

def get_dose_batch(control_matrices, leafs, mus, dose_shape, num_cp=6):
    dose = tf.zeros(dose_shape, dtype=tf.float16)
    for cp_idx in range(num_cp):
        for j in range(64): # Number of leaves
            leaf_min = tf.expand_dims(leafs[:, 0, j, cp_idx], axis=-1)  # Shape: (batch_size, 1)
            leaf_max = tf.expand_dims(leafs[:, 1, j, cp_idx], axis=-1)  # Shape: (batch_size, 1)
            for i in range(64): # Number of leaves
                idx_value = j * 64 + i + 1
                leaf_idx_pos = tf.cast(i / 64, dtype=tf.float16)
                
                condition = tf.logical_and(tf.logical_and(tf.less_equal(leaf_idx_pos, leaf_max), tf.greater_equal(leaf_idx_pos, leaf_min)), tf.equal(control_matrices[cp_idx], idx_value))
                dose += tf.where(condition, mus[..., cp_idx], 0)
    return dose

def monaco_param_to_vector(leafs, mus):
    leafs = tf.stack(leafs, -1)
    mus = tf.stack(mus, -1)

    return tf.concat([tf.reshape(leafs, [-1]), tf.reshape(mus, [-1])], -1)

def vector_to_monaco_param(vec, leaf_length=64, num_cp=6, batch_size=1):
    leafs = tf.reshape(vec[:batch_size * leaf_length * 2 * num_cp], [batch_size, 2, leaf_length, num_cp])
    mus = tf.reshape(vec[batch_size * leaf_length * 2 * num_cp:], [batch_size, 1, num_cp])
    return leafs, mus