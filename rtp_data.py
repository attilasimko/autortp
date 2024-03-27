import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from scipy.ndimage import zoom
import random
from scipy.ndimage import interpolation
import tensorflow
from natsort import natsorted 
import gc
import tensorflow as tf
from keras import backend as K
import os
from cv2 import resize
sys.path.append("../miqa-seg")
from utils import model_config



class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 structs,
                 img_size,
                 batch_size=32,
                 shuffle=True
                 ):

        self.img_size = img_size
        self.data_path = data_path
        self.structs = structs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
        self.clean_iter = 0
        self.clean_thr = 100

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError(f'The data path ({repr(data_path)}) is not a directory.')

        self.file_idx = 0
        self.file_list = os.listdir(self.data_path)
        self.indexes = np.arange(len(self.file_list))

        if (self.shuffle):
            self.file_list.sort()
        else:
            self.file_list = natsorted(self.file_list)
        
        if (self.shuffle):
            self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        if (index > len(self) - 1):
            print("Index is larger than the number of allowed batches.")
            index = index % len(self)
        
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.temp_ID = [self.file_list[k] for k in indexes]
            
        # Generate data
        i, o = self.__data_generation(self.temp_ID)
        self.clean_iter += 1

        if self.clean_iter >= self.clean_thr:
            gc.collect()
            K.clear_session()
            self.clean_iter = 0
        
        return i, o

    def on_epoch_end(self):
        if (self.shuffle):
            np.random.shuffle(self.indexes)

    #@threadsafe_generator
    def __data_generation(self, file_path_list):

        'Generates data containing batch_size samples'
        inputs = []
        outputs = []

        if (len(file_path_list) != self.batch_size):
            raise ValueError("Batch size is not equal to the number of files in the list")

        for file_path in file_path_list:
            with np.load(self.data_path + file_path) as file:
                ct = np.array(file['CT'], dtype=float)
                ct = zoom(ct, (self.img_size[0] / ct.shape[0], self.img_size[1] / ct.shape[1], self.img_size[2] / ct.shape[2]))
                ct = np.expand_dims(np.interp(ct, (ct.min(), ct.max()), (-1.0, 1.0)), 3)

                ptv = np.array(file['PTV'], dtype=float)
                ptv = zoom(ptv, (self.img_size[0] / ptv.shape[0], self.img_size[1] / ptv.shape[1], self.img_size[2] / ptv.shape[2])) > 0.5

                masks = []
                masks.append(ptv)
                aux = np.array(file['Aux'], dtype=int)
                for struct in self.structs:
                    mask = aux == self.config.maps[struct]
                    mask = zoom(mask, (self.img_size[0] / mask.shape[0], self.img_size[1] / mask.shape[1], self.img_size[2] / mask.shape[2])) > 0.5
                    masks.append(mask)
                
                
                masks = np.stack(masks, -1)

                dose = np.array(file['Dose'], dtype=float)
                dose = zoom(dose, (self.img_size[0] / dose.shape[0], self.img_size[1] / dose.shape[1], self.img_size[2] / dose.shape[2]))
                dose = np.expand_dims(dose, 3)
                
            inputs.append(np.concatenate([ct, masks[..., 0:1]], -1))
            outputs.append(np.concatenate([dose, masks], -1))
        return np.stack(inputs, 0), np.stack(outputs, 0)