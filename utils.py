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

    # indeces = y * (x + 1) * 128 + x
    # indeces = np.swapaxes(indeces, 1, 2)

    d2 = (x - center[0])**2 + (y - center[1])**2
    gaussian = np.exp(-d2 / (2 * sigma**2))
    gaussian /= np.max(gaussian)
    gaussian = np.repeat(gaussian[..., None], 128, -1)
    gaussian = np.swapaxes(gaussian, 1, 2)

    for i in range(num_cp):
        rotated_arrays.append(np.expand_dims(np.array(rotate(gaussian, i * 360 / num_cp, (0, 1), reshape=False), dtype=np.float16)[32:96, 32:96, 32:96], 0))

    return rotated_arrays