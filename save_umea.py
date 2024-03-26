import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pydicom
import numpy as np
import sys
import cv2
import nibabel as nib
from skimage import draw
import matplotlib.pyplot as plt
sys.path.append("../miqa-seg")
from utils import model_config
from model import build_unet
from tensorflow.keras.models import Model
from scipy.ndimage.interpolation import shift
import tensorflow as tf
import SimpleITK as sitk
from rt_utils import RTStructBuilder
from pathlib import Path
from scipy.ndimage import zoom
sys.path.insert(1, os.path.abspath('.'))
import rtp_utils

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/ARTP/"
base_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/DSets/ARTP/"

# Program to count islands in boolean 2D matrix
class Graph:
	def __init__(self, g):
		self.ROW = g.shape[0]
		self.COL = g.shape[1]
		self.SLC = g.shape[2]
		self.graph = g

	# A function to check if a given cell
	# (row, col, slc) can be included in DFS
	def isSafe(self, i, j, k, visited):
		# row number is in range, column number
		# is in range and value is 1
		# and not yet visited
		return (i >= 0 and i < self.ROW and
				j >= 0 and j < self.COL and
				k >= 0 and k < self.SLC and
				(visited[i, j, k] == 0) and self.graph[i, j, k])

	# A utility function to do DFS for a 2D
	# boolean matrix. It only considers
	# the 8 neighbours as adjacent vertices

	def DFS(self, i, j, k, visited, idx):
		# These arrays are used to get row and
		# column numbers of 8 neighbours
		# of a given cell
		rowNbr = [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
		colNbr = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
		slcNbr = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

		# Mark this cell as visited
		queue = []

		queue.append([i, j, k])
		while (len(queue) > 0):
			[s_i, s_j, s_k] = queue.pop(0)
			if (visited[s_i, s_j, s_k] > 0):
				continue

			visited[s_i, s_j, s_k] = idx
			for l in range(len(slcNbr)):
				if self.isSafe(s_i + rowNbr[l], s_j + colNbr[l], s_k + slcNbr[l], visited):
					queue.append([s_i + rowNbr[l], s_j + colNbr[l], s_k + slcNbr[l]])


	# The main function that returns
	# count of islands in a given boolean
	# 2D matrix

	def countIslands(self):
		# Make a bool array to mark visited cells.
		# Initially all cells are unvisited
		visited = np.zeros_like(self.graph, dtype=np.int16)

		# Initialize count as 0 and traverse
		# through the all cells of
		# given matrix
		count = 0
		for k in range(self.SLC - 1, 0, -1):
			if (np.sum(self.graph[:, :, k]) == 0):
				continue
			for i in range(self.ROW):
				if (np.sum(self.graph[i, :, k]) == 0):
					continue
				for j in range(self.COL):
					if ((visited[i, j, k] == 0) & (self.graph[i][j][k] == True)):
						if (j < 256):
							count += 1
							self.DFS(i, j, k, visited, 1)
						if (j >= 256):
							count += 1
							self.DFS(i, j, k, visited, 2)
		return visited, count
	
def resize(img):
    new_img = np.zeros((512, 512, np.shape(img)[2]))
    for i in range(np.shape(img)[2]):
        new_img[:,:,i] = cv2.resize(np.array(img[:,:,i], dtype=np.float32), (512, 512))
    
    return new_img

def get_data(path):
    img = nib.load(path).get_fdata()
    img = resize(np.flip(np.transpose(img, (1, 2, 0)), axis=2))
    return img

roi_names = []
patients = os.listdir(os.path.join(data_path))
patients = patients

num_filters = 48
config = model_config("../miqa-seg/data_evaluation/PatientLabelProcessed.csv")
seg_path = "weights/crimson_cliff_1296_hero.h5"
seg_model = build_unet(len(config.labels), 1, num_filters, 0.0, config.epsilon)
seg_model.load_weights(seg_path, skip_mismatch=True, by_name=True)
seg_model.compile(loss="mse")
seg_model.trainable = False
for l in seg_model.layers:
    l.trainable = False

for patient in patients:
	try:
		if (patients.index(patient) / len(patients) < 0.7):
			datatype = "training"
		elif (patients.index(patient) / len(patients) < 0.85):
			datatype = "validating"
		else:
			datatype = "testing"

		STACK = []
		for scan_file in os.listdir(os.path.join(data_path, patient)):
			if scan_file.__contains__("RTDOSE"):
					rtdose_path = os.path.join(data_path, patient, scan_file)
			if scan_file.__contains__("RTSTRUCT"):
					rtstruct_path = os.path.join(data_path, patient, scan_file)
			if scan_file.startswith("CT"):
					image_path = os.path.join(data_path, patient)
					data = pydicom.dcmread(os.path.join(data_path, patient, scan_file))
					STACK.append(data)

		
		doseArray, ct_stack     = [], []
		doseSpacing, ctSpacing = [], []
		doseImagePositionPatient, ctImagePositionPatientMin, ctImagePositionPatientMax = [], [], []

		dsCTs = []
		for pathCTDicom in Path(image_path).iterdir(): 
			if (pathCTDicom.name.startswith("CT")):
				dsCTs.append(pydicom.dcmread(pathCTDicom))

		ctSpacing = [float(each) for each in dsCTs[0].PixelSpacing] + [float(dsCTs[0].SliceThickness)]
		dsCTs.sort(key=lambda x: x.InstanceNumber)
		ctImagePositionPatientMin = [float(each) for each in dsCTs[0].ImagePositionPatient]
		ctImagePositionPatientMax = [float(each) for each in dsCTs[-1].ImagePositionPatient]
		for dsCT in dsCTs:
				ct_stack.append(dsCT.pixel_array * dsCT.RescaleSlope + dsCT.RescaleIntercept)
		ct_stack = np.array(ct_stack[::-1])
		ct_stack = np.clip(ct_stack, -1000, 1000)
		ct_stack /= 1000

		RTSTRUCT = RTStructBuilder.create_from(
		dicom_series_path=image_path,
		rt_struct_path=rtstruct_path
		)
		
		ptvName = [structName for structName in RTSTRUCT.get_roi_names() if "PTV" in structName][0]
		PTV = RTSTRUCT.get_roi_mask_by_name(ptvName)

		dsDose      = pydicom.dcmread(rtdose_path)
		doseArray   = dsDose.pixel_array * dsDose.DoseGridScaling
		doseSpacing = [float(each) for each in dsDose.PixelSpacing] + [float(dsDose.SliceThickness)]
		assert doseArray.shape[0] == float(dsDose.NumberOfFrames)
		doseImagePositionPatient = [float(each) for each in dsDose.ImagePositionPatient]

		doseImage = sitk.GetImageFromArray(doseArray)
		doseImage.SetSpacing(doseSpacing)
		resampler = sitk.ResampleImageFilter()
		resampler.SetOutputSpacing(ctSpacing)
		resampler.SetSize(ct_stack.shape[::-1]) # SimpleITK convention: [H,W,Slices], numpy convention: [Slices,H,W]
		resampled_image = resampler.Execute(doseImage)
		doseArrayResampled = sitk.GetArrayFromImage(resampled_image)

		dx, dy, dz = ((np.array(doseImagePositionPatient) - np.array(ctImagePositionPatientMax)) / np.array(ctSpacing)).astype(int)
		doseArrayResampled = shift(doseArrayResampled, (dz, dy, dx))

		ct_stack = np.moveaxis(ct_stack, 0, 2)
		doseArrayResampled = np.moveaxis(doseArrayResampled, 0, 2)
		Segmentations = np.ones_like(ct_stack, dtype=np.int16)
		for i in range(ct_stack.shape[2]):
			Segmentations[..., i] = np.argmax(seg_model.predict_on_batch(np.expand_dims(ct_stack[..., i:i+1], 0)), -1)

		mass = np.array((PTV * np.mgrid[0:PTV.shape[0], 0:PTV.shape[1], 0:PTV.shape[2]]).sum(1).sum(1).sum(1)/PTV.sum(), dtype=np.int16)

		offset = [(256 - mass[0]) * 2, (256 - mass[1]) * 2, 0]
		pad = ((offset[0] if offset[0] > 0 else 0, -offset[0] if offset[0] < 0 else 0), (offset[1] if offset[1] > 0 else 0, -offset[1] if offset[1] < 0 else 0), (0, 0))
		# PTV = np.pad(PTV, pad, mode='constant', constant_values=0)
		# ct_stack = np.pad(ct_stack, pad, mode='constant', constant_values=-1000)
		# doseArrayResampled = np.pad(doseArrayResampled, pad, mode='constant', constant_values=0)
		# Segmentations = np.pad(Segmentations, pad, mode='constant', constant_values=0) 

		# ct_stack = ct_stack[mass[0]-64:mass[0]+64, mass[1]-64:mass[1]+64, mass[2]-64:mass[2]+64]
		# Kidney = a[mass[0]-64:mass[0]+64, mass[1]-64:mass[1]+64, mass[2]-64:mass[2]+64]
		orig_size = (256, 256, 128)
		ct_stack = zoom(ct_stack, (orig_size[0] / ct_stack.shape[0], orig_size[1] / ct_stack.shape[1], orig_size[2] / ct_stack.shape[2]))
		PTV = zoom(PTV, (orig_size[0] / PTV.shape[0], orig_size[1] / PTV.shape[1], orig_size[2] / PTV.shape[2]), order=0)
		doseArrayResampled = zoom(doseArrayResampled, (orig_size[0] / doseArrayResampled.shape[0], orig_size[1] / doseArrayResampled.shape[1], orig_size[2] / doseArrayResampled.shape[2]))
		Segmentations = zoom(Segmentations, (orig_size[0] / Segmentations.shape[0], orig_size[1] / Segmentations.shape[1], orig_size[2] / Segmentations.shape[2]), order=0)

		np.savez_compressed("/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/DSets/ARTP/" + datatype + "/" + patient,
												CT = np.array(np.interp(ct_stack, (-1, 1), (0, 255)), dtype=np.int16),
												PTV = np.array(PTV, dtype=bool),
												Dose = np.array(doseArrayResampled, dtype=np.float32),
												Aux = np.array(Segmentations, dtype=np.int16)
		)
	except Exception as e:
			print("Error in patient: ", patient + " - " + str(e))