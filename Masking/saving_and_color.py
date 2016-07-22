import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import median_filter
from PyQt4 import QtGui, QtCore
import cv2
import tifffile
import os
import math
import re
import time
import arrayfire as af


def saveMIP(validityMap, mappedMip, data, size): #####################fix this. what is this even doing?
    dialog = QtGui.QFileDialog()
    filename = str(dialog.getSaveFileName(filter=QtCore.QString('Images (*.tif)')))
    if not filename:  # no filename was created
        return
    if type(validityMap) is bool:
        tifffile.imsave(filename, data)
        return
    height, width, numcolors = data.shape
    # initiate a progress bar
    bar = QtGui.QProgressBar()
    bar.setWindowTitle(QtCore.QString('Saving Masked MIP Image...'))
    bar.setWindowModality(QtCore.Qt.WindowModal)
    bar.resize(size, size / 20)
    bar.move(size, size)
    currentProgress = 0
    bar.setMaximum(height + (2 * (height / 20)))
    bar.show()
    QtGui.QApplication.processEvents()
    # start processing:
    data = data.reshape((height * width), numcolors)
    black = [0 for x in xrange(0, numcolors)]
    indices = []
    for y in xrange(0, height):  # apply validityMap (mask) to entire image
        yshift = y * height
        for x in xrange(0, width):
            cx, cy, cv = mappedMip[0][y][x], mappedMip[1][y][x], mappedMip[2][y][x]
            if not validityMap[cv][cx][cy]:
                indices.append((yshift + x))
        currentProgress += 1
        bar.setValue(currentProgress)
        QtGui.QApplication.processEvents()
    data[indices] = black  # set pixels to black
    currentProgress += height / 20
    bar.setValue(currentProgress)
    QtGui.QApplication.processEvents()
    data = data.reshape(height, width, numcolors)  # reshape back to normal
    tifffile.imsave(filename, data)
    currentProgress += height / 20
    bar.setValue(currentProgress)
    QtGui.QApplication.processEvents()


def applyToStack(xyvMaps, size, opendirectory, boundsinclude, colorMode):
    # initiate a progress bar
    bar = QtGui.QProgressBar()
    bar.setWindowTitle(QtCore.QString('Applying Mask to Stack...'))
    bar.setWindowModality(QtCore.Qt.WindowModal)
    bar.resize((size * 2), size / 20)
    bar.move(size, size)
    files = []
    fileIndices = []
    number = re.compile(r'\d+')
    for file in os.listdir(opendirectory):
        if file.endswith(".tif") and 'mip' not in file:
            files.append(file)
            numbers = re.findall(number, file)
            if len(numbers) != 1:
                error = QtGui.QMessageBox()
                error.setText(QtCore.QString('Error! One or more of the input'
                    ' files contains 0 or >1 numbers in the filename, which'
                    ' are necessary for indexing Tifs by their z-layer. '
                    'Aborting...'))
                error.exec_()
                return
            fileIndices.append(int(numbers[0]))
    if len(files) == 0:
        error = QtGui.QMessageBox()
        error.setText(QtCore.QString('Error! There are not tif files in this '
            'folder.'))
        error.exec_()
        return
    files = [y for (x, y) in sorted(zip(fileIndices, files))]
    # update progress bar configuration
    bar.setMaximum(((len(files) + 2) * len(xyvMaps)))
    progress = 0
    bar.show()
    QtGui.QApplication.processEvents()
    # make the save directories
    saveDirectory = opendirectory + '/maskedTIFs'
    saveDilatedDir = saveDirectory + '/Dilated/'
    saveUndilatedDir = saveDirectory + '/Undilated/'
    saveDilatedStackDir = saveDilatedDir + 'Stack/'
    saveUndilatedStackDir = saveUndilatedDir + 'Stack/'
    for path in [saveDirectory, saveDilatedDir, saveUndilatedDir,
                 saveDilatedStackDir, saveUndilatedStackDir]:
        if not os.path.exists(path):
            os.makedirs(path)
    numpystacks = [[] for x in xrange(0, len(xyvMaps))]  # [Color[Stack]]
    originalStack = []
    radius = size / 2
    for filenum, file in enumerate(files):
        bar.setWindowTitle(QtCore.QString('Applying Mask to Stack...to Z-Layer %d Color 1' % (filenum+1)))
        QtGui.QApplication.processEvents()
        # open the array
        with tifffile.TIFFfile((opendirectory + '/' + file)) as tif:
            original = tif.asarray()
        if original.shape[2] == 4:
            original = original[:, :, 0:3]
        originalStack.append(original)
        assert (original.dtype != np.uint8)
        if boundsinclude == [[[0, 127, 255], [0, 127, 255], [0, 127, 255]], [True, True, True]]:
            rgb = original.copy()
            rgb /= 256
            rgb = rgb.astype(np.uint8)
        else:
            [bounds, include] = boundsinclude
            rgb = rgbCorrection(original.astype(np.float32), bounds, False, include)  # apply correction to rgb
        if colorMode != 'rgb':
            xyv = rgbtoxyv(rgb, radius, colorMode)
        height, width, numcolors = original.shape
        original = original.reshape((height * width), numcolors)
        if numcolors == 3:
            black = [0, 0, 0]
        elif numcolors == 4:
            black = [0, 0, 0, 255]
        for color, map in enumerate(xyvMaps):
            # process the array
            cropped = original.copy()  # what is to be saved
            indices = []
            for py in xrange(0, height):
                yshift = py * width
                for px in xrange(0, width):
                    [x, y, v] = xyv[0][py][px], xyv[1][py][px], xyv[2][py][px]
                    if not map[v][x][y]:
                        indices.append((yshift + px))
            cropped[indices] = black  # set pixels to black
            cropped = cropped.reshape(height, width, numcolors)  # reshape back to normal
            # save the array as tif
            tifffile.imsave((saveUndilatedStackDir + ('Color%d_' % (color+1)) + file), cropped)
            # save the numpy array into numpystack
            numpystacks[color].append(cropped.copy())
            # update progressbar
            progress += 1
            bar.setValue(progress)
            bar.setWindowTitle(QtCore.QString('Applying Mask to Stack...to Z-Layer %d Color %d' % (filenum+1, color+2)))
            QtGui.QApplication.processEvents()
    for color in xrange(0, len(xyvMaps)):
        bar.setWindowTitle(QtCore.QString('Creating MIP for Color %d' % (color+1)))
        QtGui.QApplication.processEvents()
        mip = np.maximum.reduce(numpystacks[color])
        tifffile.imsave((saveUndilatedDir + ('MIP_Undilated_Color%d' % (color+1)) + '.tif'), mip)
        progress += 1
        bar.setValue(progress)
        QtGui.QApplication.processEvents()
    dim3bools = []
    for color in numpystacks:
        rgb3D = np.array(color)
        bool3D = rgb3D[:, :, :, 0] > 0
        dim3bools.append(bool3D)
    dilationStruct = np.array([[[False, False, False], [False, True, False], [False, False, False]],  # z = -1
                              [[False, True, False], [True, True, True], [False, True, False]],  # z = 0
                              [[False, False, False], [False, True, False], [False, False, False]]])  # z = -1
    for color, image in enumerate(dim3bools):
        bar.setWindowTitle(QtCore.QString('Dilating Color %d and Saving Stack with MIP' % (color + 1)))
        QtGui.QApplication.processEvents()
        dilated3DMask = ndimage.binary_dilation(image, structure=dilationStruct)
        # the following median filter is in beta #########################
        for x in xrange(0, 2):
            dilated3DMask = median_filter(dilated3DMask, size=(3, 3, 3))  # this is in beta
            dilated3DMask = ndimage.binary_dilation(dilated3DMask, structure=dilationStruct)
        dilated3DMask = ndimage.binary_dilation(dilated3DMask, structure=dilationStruct)
        # end beta #######################################################
        dilated3DMask = np.expand_dims(dilated3DMask, axis=3)
        dilated3DMask = np.repeat(dilated3DMask, originalStack[0].shape[2], axis=3)
        # dilated3DMask = dilated3DMask.astype(np.uint16)
        dilatedStack = []
        for layer, file in enumerate(files):
            dilatedImage = dilated3DMask[layer] * originalStack[layer]
            dilatedImage = median_filter(dilatedImage, size=(3, 3, 1))  # this is in beta
            # dilatedImage = cv2.bitwise_and(originalStack[layer], originalStack[layer], mask=dilated3DMask[layer])  # dilated3DMask[layer] * originalStack[layer]
            dilatedStack.append(dilatedImage)
            tifffile.imsave((saveDilatedStackDir + ('Dilated_Color%d_' % (color+1)) + file), dilatedImage)
        dilatedMip = np.maximum.reduce(dilatedStack)
        # print 'comparison of the 2 MIPs:', (dilatedMip==mip).all()
        tifffile.imsave((saveDilatedDir + ('MIP_Dilated_Color%d' % (color+1)) + '.tif'), dilatedMip)
        progress += 1
        bar.setValue(progress)
        QtGui.QApplication.processEvents()

def rgbCorrection(img, bounds, gpuMode, include):
    # converts a float32 to 8uint arr
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    for c, color in enumerate([r, g, b]):
        height, width = color.shape[0], color.shape[1]
        if not include[c]:
            img[:, :, c] = np.zeros((height, width))
            continue
        [i, m, f] = bounds[c]
        i, m, f = i * 256, m * 256, f * 256
        maximum = 65535  # in 16-bit unsigned
        color[color < i] = i
        color[color > f] = f
        predictedMid = (i + f) / 2
        if gpuMode:
            color = color.reshape((width * height))
        overmid = color.copy()
        overmidmask = (overmid >= m)
        if gpuMode:
            overmid = af.interop.np_to_af_array(overmid)
            for ii in af.ParallelRange(width*height):
                overmid[ii] -= m
                overmid[ii] *= (float(maximum - predictedMid) / (f - m))
                overmid[ii] += predictedMid
            overmid = np.array(overmid)
            overmid = overmid.reshape(height, width)
        else:
            overmid = ((overmid-m) * (float(maximum - predictedMid) / (f - m))) + predictedMid
        overmid *= overmidmask
        undermid = color
        undermidmask = (undermid < m)
        if gpuMode:
            undermid = af.interop.np_to_af_array(undermid)
            for ii in af.ParallelRange(width*height):
                undermid[ii] -= i
                undermid[ii] *= (float(predictedMid - i) / (m - i))
            undermid = np.array(undermid)
            undermid = undermid.reshape(height, width)
        else:
            undermid -= i
            undermid *= (float(predictedMid - i) / (m - i))
        undermid *= undermidmask
        color = undermid + overmid
        img[:, :, c] = color
    img /= 256
    img = img.astype(np.uint8)
    return img

def rgbtoxyv(rgb, radius, colorMode):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)
    # hsv = [0-255, 0-255, 0-255]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h = h.astype(np.float32)
    h *= 0.024639942381096416  # convert h from 255 to radians
    if colorMode == 'hsvI':
        radius *= 0.8
        '''if radius > 128:
            radius = 128'''
    elif colorMode == 'hsv':
        s = 255 - s
    s = s.astype(np.float32)
    s /= 255
    x = (s * np.cos(h) + 1) * radius  # set x
    y = (1 - s * np.sin(h)) * radius  # set y
    realSideMax = radius * 2 - 1
    x = x.astype(np.uint16)
    x[x > realSideMax] = realSideMax
    y = y.astype(np.uint16)
    y[y > realSideMax] = realSideMax
    #xyv = np.zeros((hsv.shape[0], hsv.shape[1], 3))
    #xyv[:, :, 0], xyv[:, :, 1], xyv[:, :, 2] = x, y, v
    xyv = [x, y, v]
    return xyv


def hsvtoxyv(hsv, radius):
    '''
    :param hsv: list of 3 ints: hsv color, h:[0, 2pi], s:[0, 1], v:[0, 255]
    :param radius: int: defines the radius of colorspaceview
    :return: xyv: list of 3 ints: xyv color according to radius,
    x:[0, radius*2], y:[0, radius*2], v:[0, 255]
    '''
    # radius of cylinder in colorSpace
    [h, s, v] = hsv
    h *= math.pi / 180  # convert h from degrees to radians
    s = 1 - s
    x = int((s * math.cos(h) + 1) * radius)  # set x
    y = int((1 - s * math.sin(h)) * radius)  # set y
    if x == radius * 2:
        x -= 1
    if y == radius * 2:
        y -= 1
    return [x, y, v]

def rgbtohsv(rgb):
    '''
    :param rgb: list of 3 ints: rgb color, r:[0, 255], g:[0, 255], g:[0, 255]
    :return: hsv: list of 3 ints: hsv color, h:[0, 360], s:[0, 1], v:[0, 255]
    '''
    [r, g, b] = rgb
    v = max(rgb)
    if v == 0:
        return [0, 0, 0]
    delta = v - min(rgb)
    s = delta / v
    if delta == 0:
        return [0, s, int(v * 255)]
    if v == r:
        h = 60 * (((g - b) / delta) % 6)
    elif v == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    return [h, s, int(v * 255)]

def hsv2rgb(hsv):
    '''
    :param hsv: list of 3 ints: hsv color, h:[0, 2pi], s:[0, 1], v:[0, 255]
    :return: rgb: list of 3 ints: rgb color, r:[0, 255], g:[0, 255], g:[0, 255]
    '''
    [h, s, v] = hsv
    v = float(v) / 255.0
    hn = (h * 3) / math.pi
    i = int(hn)
    if i == 6:
        i = 5
    x = v * (1 - s)
    y = v * (1 - s * (hn - i))
    z = v * (1 - s * (1 - hn + i))
    if i == 0:
        (r, g, b) = (v, z, x)
    elif i == 1:
        (r, g, b) = (y, v, x)
    elif i == 2:
        (r, g, b) = (x, v, z)
    elif i == 3:
        (r, g, b) = (x, y, v)
    elif i == 4:
        (r, g, b) = (z, x, v)
    else:
        (r, g, b) = (v, x, y)
    return (int(r * 255), int(g * 255), int(b * 255))
