import numpy as np
import matplotlib.pyplot as plt
import os
from cv2 import resize


def generate_data(seg_model, batch_size=1, idx=None):
    data_path = "data/"
    data_list = os.listdir(data_path)
    data = []
    dose = []
    dose_scale = 2.4
    shape = (64, 64, 64)
    ct = np.ones(shape, dtype=float)
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    for batch in range(batch_size):
        if (idx is None):
            idx = 0 # np.random.randint(len(data_list))
        segmentations = []
        with np.load(data_path + data_list[idx]) as file:
            ct = np.array(file['CT'], dtype=float)
            ct = np.expand_dims(np.interp(ct, (ct.min(), ct.max()), (-1.0, 1.0)), 3)
            mask = np.expand_dims(np.array(file['Kidney'], dtype=float), axis=3)
        for slc_idx in range(ct.shape[2]):
            slc = np.expand_dims(resize(ct[..., slc_idx, 0], (512, 512)), (0, 3))
            slc = seg_model.predict_on_batch(slc)
            structs = []
            for struct_idx in range(len(slc)):
                structs.append(resize(slc[struct_idx][0, ..., 0], (256, 256)))
            segmentations.append(np.stack(structs, -1))
        segmentations = np.stack(segmentations, axis=2)

        # sigma = 10 # np.random.randint(5, 15)
        # center = [32, 32, 32] # np.random.randint(12, 52)]
        # d2 = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2
        # gaussian = np.exp(-d2 / (2 * sigma**2))
        # gaussian /= np.max(gaussian)
        # mask = np.array(gaussian > 0.5, dtype=float)
            
        data.append(np.concatenate([ct, mask, segmentations], -1))
        dose.append(dose_scale*mask)
    return np.stack(data, 0).astype(np.float32), np.stack(dose, 0).astype(np.float32)