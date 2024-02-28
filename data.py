import numpy as np
import matplotlib.pyplot as plt

def generate_data(batch_size=1, center=None, sigma=None):
    data = []
    dose = []
    dose_scale = 1.2
    shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    for batch in range(batch_size):
        if sigma is None:
            sigma = np.random.randint(3, 30)
        if center is None:
            center = np.random.randint(0, 64, size=2)
        d2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        gaussian = np.exp(-d2 / (2 * sigma**2))
        gaussian /= np.max(gaussian)
        mask = np.array(gaussian > 0.5, dtype=float)
        data.append(np.stack([ct, mask], axis=-1))
        dose.append(np.expand_dims(dose_scale*mask, -1))
    return np.stack(data, 0), np.stack(dose, 0)