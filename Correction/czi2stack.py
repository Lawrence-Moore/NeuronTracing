from PyQt4 import QtGui, QtCore
import sys
from image_normalization import read_czi_file
import numpy as np
import os
import tifffile

app = QtGui.QApplication(sys.argv)
dialog = QtGui.QFileDialog()
filename = str(dialog.getOpenFileName(filter=QtCore.QString('CZI File (*.czi)')))
if not filename:  # user pressed cancel
    print 'no file was selected for opening. aborting.'
    quit()

print 'creating save directory'
savedirectory = filename[:-4]  # get folder name from filename
if not os.path.exists(savedirectory):
    os.makedirs(savedirectory)

print 'reading czi file'
originalData = read_czi_file(filename)

print 'creating mip'
mip = np.maximum.reduce(originalData)

print 'saving mip'
tifffile.imsave((savedirectory + '/mip.tif'), mip)

print 'saving individual layers'
for i, zLayerImg in enumerate(originalData):
    filename = savedirectory + ('/original_z_%d.tif' % i)
    tifffile.imsave(filename, zLayerImg)

print 'job completed successfully. quitting'
