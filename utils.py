import numpy as np
from scipy.ndimage import rotate

def get_monaco_projections(num_cp):
    rotated_arrays = []
    shape = (128, 128, 128)
    new_shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    sigma = 10
    center = (64, 64)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

    indeces = y * shape[0] + x
    indeces = np.repeat(indeces[..., None], 128, -1)
    indeces = np.swapaxes(indeces, 1, 2)

    # d2 = (x - center[0])**2 + (y - center[1])**2
    # gaussian = np.exp(-d2 / (2 * sigma**2))
    # gaussian /= np.max(gaussian)
    # gaussian = np.repeat(gaussian[..., None], 128, -1)
    # gaussian = np.swapaxes(gaussian, 1, 2)

    for angle_idx in range(num_cp):
        array = np.expand_dims(rotate(indeces, angle_idx * 360 / num_cp, (0, 1), reshape=False, order=0)[32:96, 32:96, 32:96], 0)
        array_replica = - np.ones_like(array)
        idx = 0
        for i in range(32, 96):
            for j in range(32, 96):
                array_replica[array == (j * shape[0] + i)] = idx
                idx += 1
        rotated_arrays.append(array_replica)

    return rotated_arrays