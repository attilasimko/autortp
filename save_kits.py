import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pydicom
import numpy as np
import cv2
import nibabel as nib
from skimage import draw
import matplotlib.pyplot as plt
import SimpleITK as sitk
from rt_utils import RTStructBuilder
import sys
from scipy.ndimage import zoom
sys.path.insert(1, os.path.abspath('.'))
import utils

data_path = "/mnt/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/kits19/original/"
base_path = "/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/ARTP/"

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

for patient in patients[:10]:
    try:
        ct_stack = get_data(data_path + patient + "/imaging.nii.gz")
        # ct_stack = np.interp(ct_stack, (-1, 1), (0, 255))

        segmentation = get_data(data_path + patient + "/segmentation.nii.gz").astype(np.uint8)
        kidney = segmentation == 1
        # tumor = segmentation == 2
		
        g = Graph(kidney)
        labels, count = g.countIslands()
		
        Kidney = np.array(labels == 1, dtype=float)
        mass = np.array((Kidney * np.mgrid[0:Kidney.shape[0], 0:Kidney.shape[1], 0:Kidney.shape[2]]).sum(1).sum(1).sum(1)/Kidney.sum(), dtype=np.int16)
	
        offset = [(256 - mass[0]) * 2, (256 - mass[1]) * 2, 0]
        pad = ((offset[0] if offset[0] > 0 else 0, -offset[0] if offset[0] < 0 else 0), (offset[1] if offset[1] > 0 else 0, -offset[1] if offset[1] < 0 else 0), (0, 0))
        Kidney = np.pad(Kidney, pad, mode='constant', constant_values=0)
        ct_stack = np.pad(ct_stack, pad, mode='constant', constant_values=-1000)
		
        # ct_stack = ct_stack[mass[0]-64:mass[0]+64, mass[1]-64:mass[1]+64, mass[2]-64:mass[2]+64]
        # Kidney = a[mass[0]-64:mass[0]+64, mass[1]-64:mass[1]+64, mass[2]-64:mass[2]+64]

        ct_stack = zoom(ct_stack, (64 / ct_stack.shape[0], 64 / ct_stack.shape[1], 64 / ct_stack.shape[2]))
        Kidney = zoom(Kidney, (64 / Kidney.shape[0], 64 / Kidney.shape[1], 64 / Kidney.shape[2])) > 0.2
        
        np.savez_compressed("data/kits19_" + patient,
                            CT = np.array(np.interp(ct_stack, (-1000, 1000), (0, 255)), dtype=np.int16),
                            Kidney = np.array(Kidney, dtype=bool)
        )
    except Exception as e:
        print("Error in patient: ", patient + " - " + str(e))