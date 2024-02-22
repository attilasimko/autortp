import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    sigma = np.random.randint(5, 20)
    center = np.random.randint(0, 64, size=3)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    d2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
    gaussian = np.exp(-d2 / (2 * sigma**2))
    gaussian /= np.max(gaussian)
    mask = np.array(gaussian > 0.5, dtype=float)
    data = np.stack([ct - 1 + 0.1 * mask, mask], axis=-1)
    return data[None, ...], 5 * mask[None, ...]