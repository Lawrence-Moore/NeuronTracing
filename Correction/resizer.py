import os
from tifffile import imread, imsave
import cv2
from PyQt4 import QtGui, QtCore
import sys
import numpy as np

compression = 4

app = QtGui.QApplication(sys.argv)
dialog = QtGui.QFileDialog()
opendirectory = str(dialog.getExistingDirectory())
if not opendirectory:  # user pressed cancel
    print 'no directory was selected for opening. aborting.'
    quit()
print opendirectory

print 'gathering files from open directory'
file_names = []
for file in os.listdir(opendirectory):
	if file.endswith('.tif'):
		file_names.append(file)

print 'create save directory'

savedirectory = opendirectory + '/resized'
if not os.path.exists(savedirectory):
	os.makedirs(savedirectory)

print 'resizing and resaving tifs'

for file_name in file_names:
	print 'processing tif:', file_name
	save_file_name = savedirectory + '/' + file_name
	arr = imread(opendirectory + '/' +file_name)
	arr = cv2.resize(arr, (arr.shape[0] / compression, arr.shape[1] / compression))
	if len(arr.shape) == 3:
		arr = np.mean(arr, axis=2).astype(np.uint16)
		# arr = np.repeat(np.expand_dims(np.mean(arr, axis=2), axis=2), 3, axis=2).astype(np.uint16)
	imsave(save_file_name, arr)
