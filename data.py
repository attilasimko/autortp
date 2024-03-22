import numpy as np
import matplotlib.pyplot as plt
import os


def generate_data(batch_size=1, idx=None):
    data_path = "data/"
    data_list = os.listdir(data_path)
    data = []
    dose = []
    dose_scale = 1.2
    shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    for batch in range(batch_size):
        if (idx is None):
            idx = np.random.randint(len(data_list))
        with np.load(data_path + data_list[idx]) as file:
            ct = np.array(file['CT'], dtype=float)
            ct = np.interp(ct, (ct.min(), ct.max()), (-1.0, 1.0))
            # mask = np.array(file['Kidney'], dtype=float)


        sigma = np.random.randint(5, 15)
        if center is None:
            center = [32, 32, np.random.randint(12, 52)]
        d2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        gaussian = np.exp(-d2 / (2 * sigma**2))
        gaussian /= np.max(gaussian)
        mask = np.array(gaussian > 0.5, dtype=float)
            
        data.append(np.expand_dims(np.stack([ct, mask], axis=-1), axis=0))
        dose.append(np.expand_dims(np.expand_dims(dose_scale*mask, axis=0), axis=-1))
    return np.concatenate(data, 0).astype(np.float32), np.concatenate(dose, 0).astype(np.float32)