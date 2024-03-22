import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import zoom


def generate_data(batch_size=1, idx=None):
    data_path = "data/"
    data_list = os.listdir(data_path)
    if (idx is None):
        idx = np.random.randint(len(data_list))
    data = []
    dose = []
    dose_scale = 1.2
    shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    for batch in range(batch_size):
        with np.load(data_path + data_list[idx]) as file:
            ct = np.array(file['CT'], dtype=float)
            ct = zoom(ct, (64 / ct.shape[0], 64 / ct.shape[1], 64 / ct.shape[2]))
            ct = np.interp(ct, (ct.min(), ct.max()), (-1.0, 1.0))
            mask = file['Kidney_L']
            mask = zoom(mask, (64 / mask.shape[0], 64 / mask.shape[1], 64 / mask.shape[2]))
            mask = np.array(mask > 0, dtype=float)
            
        data.append(np.expand_dims(np.stack([ct, mask], axis=-1), axis=0))
        dose.append(np.expand_dims(np.expand_dims(dose_scale*mask, axis=0), axis=-1))
    return np.concatenate(data, 0).astype(np.float32), np.concatenate(dose, 0).astype(np.float32)