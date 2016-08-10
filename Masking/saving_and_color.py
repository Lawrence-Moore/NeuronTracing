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
import color_chooser


def applyToStack(maps, size, opendirectory, boundsinclude, colorMode, gpuMode, grayScale=False):
    print 'applying to stack in gpu mode', gpuMode
    if colorMode == 'rgbClusters':
        [merges, maps, full] = maps
    # initiate a progress bar
    bar = QtGui.QProgressBar()
    bar.setWindowTitle(QtCore.QString('Applying Mask to Stack...'))
    bar.setWindowModality(QtCore.Qt.WindowModal)
    bar.resize((size * 2), size / 20)
    bar.move(size, size)
    files, opendirectory = getFiles(opendirectory)
    if not files:
        return
    # update progress bar configuration
    mipProgress, dilationProgress = 3, 10
    bar.setMaximum(((len(files) + (mipProgress + dilationProgress)) * len(maps)))
    progress = 0
    bar.show()
    QtGui.QApplication.processEvents()
    # make the save directories
    saveDirectory = opendirectory + '/maskedTIFs'
    saveDilatedDir = saveDirectory + '/Dilated/'
    saveUndilatedDir = saveDirectory + '/Undilated/'
    # saveDilatedStackDir = saveDilatedDir + 'Stack/'
    # saveUndilatedStackDir = saveUndilatedDir + 'Stack/'
    numMaps = len(maps)
    for path in [saveDirectory, saveDilatedDir, saveUndilatedDir]:
        if not os.path.exists(path):
            os.makedirs(path)
    for dir in [saveDilatedDir, saveUndilatedDir]:
        for color in xrange(0, numMaps):
            path = dir + ('Color%d/' % (color+1))
            if not os.path.exists(path):
                os.makedirs(path)
    numpystacks = [[] for x in xrange(0, numMaps)]  # [Color[Stack]]
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
        assert (original.dtype != np.uint8)
        if boundsinclude == [[[0, 127, 255], [0, 127, 255], [0, 127, 255]], [True, True, True]]:
            rgb = original.copy()
            rgb /= 256
            rgb = rgb.astype(np.uint8)
        else:   # apply correction to rgb
            [bounds, include] = boundsinclude
            rgb, original = rgbCorrection(original.astype(np.float32), bounds, False, include, both=True)
        originalStack.append(original)
        if colorMode[0:3] != 'rgb':  # it's not 'rgb' or 'rgbClusters'
            if gpuMode:
                mappedImage = rgb2xyv(rgb, radius, colorMode, only='Numpy')
            else:
                mappedImage = rgb2xyv(rgb, radius, colorMode)
        else:
            mappedImage = rgb
        height, width, numcolors = original.shape
        if colorMode[0:11] != 'rgbClusters':
            original = original.reshape((height * width), numcolors)
        if numcolors == 3:
            black = [0, 0, 0]
        elif numcolors == 4:
            black = [0, 0, 0, 255]
        for color, map in enumerate(maps):
            # process the array
            if colorMode == 'rgbClusters':
                # maps = self.rgbList
                cropped = original.copy()
                mipFloat = mappedImage.astype(np.float32)
                raxis, gaxis, baxis = np.split(mipFloat, 3, axis=2)
                stack = merged2Originals(map, full, merges)
                fullMask = False
                for [r, g, b] in stack:
                    mask = True
                    dist = ((raxis - r) ** 2 + (gaxis - g) ** 2 + (baxis - b) ** 2)
                    for [ar, ag, ab] in full:
                        if [ar, ag, ab] == [r, g, b]:  # don't compare with one's self
                            continue
                        adist = ((raxis - ar) ** 2 + (gaxis - ag) ** 2 + (baxis - ab) ** 2)
                        mask *= dist <= adist
                    fullMask += mask
                fullMask = np.repeat(fullMask, numcolors, axis=2)
                cropped[~fullMask] = 0
            elif colorMode == 'rgbClusters3D':  # warning, this will save 8-bit images
                mask = map[filenum]
                mask = mask.astype(bool)
                mask = ~mask
                cropped = original.copy()
                cropped[mask] = 0
            else:
                if gpuMode:
                    cropped = fullGPUMask(original, size, map, mappedImage)
                else:
                    cropped = original.copy()  # what is to be saved
                    indices = []
                    for py in xrange(0, height):
                        yshift = py * width
                        for px in xrange(0, width):
                            [a, b, c] = mappedImage[py][px]
                            if not map[c, a, b]:
                                indices.append((yshift + px))
                    cropped[indices] = black  # set pixels to black
                    cropped = cropped.reshape(height, width, numcolors)  # reshape back to normal
            # save the numpy array into numpystack
            numpystacks[color].append(cropped.copy())
            # save the array as tif
            cropped = makeGrayScale(cropped, grayScale)
            tifffile.imsave((saveUndilatedDir + ('Color%d/' % (color+1)) + file), cropped)
            # update progressbar
            progress += 1
            bar.setValue(progress)
            bar.setWindowTitle(QtCore.QString('Applying Mask to Stack...to Z-Layer %d Color %d' % (filenum+1, color+2)))
            QtGui.QApplication.processEvents()
    for color in xrange(0, numMaps):
        bar.setWindowTitle(QtCore.QString('Creating MIP for Color %d' % (color+1)))
        QtGui.QApplication.processEvents()
        mip = np.maximum.reduce(numpystacks[color])
        mip = makeGrayScale(mip, grayScale)
        tifffile.imsave((saveUndilatedDir + ('MIP_Undilated_Color%d' % (color+1)) + '.tif'), mip)
        progress += mipProgress
        bar.setValue(progress)
        QtGui.QApplication.processEvents()
    dim3bools = []
    for color in numpystacks:
        rgb3D = np.array(color)
        bool3D = (rgb3D[:, :, :, 0] > 0) + (rgb3D[:, :, :, 1] > 0) + (rgb3D[:, :, :, 2] > 0)
        dim3bools.append(bool3D)
    dilationStruct = np.array([[[False, False, False], [False, True, False], [False, False, False]],  # z = -1
                              [[False, True, False], [True, True, True], [False, True, False]],  # z = 0
                              [[False, False, False], [False, True, False], [False, False, False]]])  # z = -1
    for color, image in enumerate(dim3bools):
        bar.setWindowTitle(QtCore.QString('Dilating Color %d and Saving Stack with MIP' % (color + 1)))
        QtGui.QApplication.processEvents()
        dilated3DMask = ndimage.binary_dilation(image, structure=dilationStruct)
        '''# the following median filter is in beta #########################
        for x in xrange(0, 2):
            dilated3DMask = median_filter(dilated3DMask, size=(3, 3, 3))  # this is in beta
            dilated3DMask = ndimage.binary_dilation(dilated3DMask, structure=dilationStruct)
        dilated3DMask = ndimage.binary_dilation(dilated3DMask, structure=dilationStruct)
        # end beta #######################################################'''
        dilated3DMask = np.expand_dims(dilated3DMask, axis=3)
        dilated3DMask = np.repeat(dilated3DMask, originalStack[0].shape[2], axis=3)
        # dilated3DMask = dilated3DMask.astype(np.uint16)
        dilatedStack = []
        for layer, file in enumerate(files):
            dilatedImage = dilated3DMask[layer] * originalStack[layer]
            # dilatedImage = median_filter(dilatedImage, size=(3, 3, 1))  # this is in beta
            # dilatedImage = cv2.bitwise_and(originalStack[layer], originalStack[layer], mask=dilated3DMask[layer])  # dilated3DMask[layer] * originalStack[layer]
            dilatedStack.append(dilatedImage)
            dilatedImage = makeGrayScale(dilatedImage, grayScale)
            tifffile.imsave((saveDilatedDir + ('Color%d/' % (color+1)) + file), dilatedImage)
        dilatedMip = np.maximum.reduce(dilatedStack)
        # print 'comparison of the 2 MIPs:', (dilatedMip==mip).all()
        dilatedMip = makeGrayScale(dilatedMip, grayScale)
        tifffile.imsave((saveDilatedDir + ('MIP_Dilated_Color%d' % (color+1)) + '.tif'), dilatedMip)
        progress += dilationProgress
        bar.setValue(progress)
        QtGui.QApplication.processEvents()

def makeGrayScale(img, grayScale):
    # works on 2D images only (8bit or 16bit)
    if not grayScale:
        return img
    full16 = False
    if img.dtype == np.uint16:
        full16 = True
    img = np.mean(img, axis=2)
    if full16:
        img = img.astype(np.uint16)
    else:
        img = img.astype(np.uint8)
    return img

def merged2Originals(rgb, rgbList, merges):
    if type(rgb[0]) is list:
        stack = rgb
    else:
        stack = [rgb]
    def search(stack):  # a recursive search
        newstack = []
        for rgb in stack:
            if rgb not in rgbList:
                found = False
                for [mc, mcs] in merges:
                    if mc == rgb:
                        found = True
                        break
                if found:
                    newstack += search(mcs)
                else:
                    print 'Error! Recursive search could not find where merged', rgb,
                    print 'originated from. Aborting...'
                    return
            else:
                newstack += [rgb]
        return newstack
    return search(stack)

def getStack(boundsinclude, full16Bit=False, withDir=False):
    dialog = QtGui.QFileDialog()
    opendirectory = str(dialog.getExistingDirectory())
    if opendirectory == '':
        return False
    files, opendirectory = getFiles(opendirectory)
    if not files:
        return False
    layers = []
    for file in files:
        with tifffile.TIFFfile((opendirectory + '/' + file)) as tif:
            original = tif.asarray()
        if original.shape[2] == 4:
            original = original[:, :, 0:3]
        if boundsinclude != [[[0, 127, 255], [0, 127, 255], [0, 127, 255]], [True, True, True]]:
            print 'undergoing image correction'
            [bounds, include] = boundsinclude
            if original.dtype == np.uint8:
                original = original.astype(np.float32)
                original *= 256
            else:
                original = original.astype(np.float32)
            eightbit, fullbit = rgbCorrection(original, bounds, False, include,
                                          both=True)
            if full16Bit:
               original = fullbit
            else:
                original = eightbit
        else:
            if original.dtype == np.uint8 and full16Bit:
                print 'Warning! Expected 16-bit images. Continuing by converting to 16bit...'
                original = original.astype(np.uint16)
                original *= 256
            elif original.dtype == np.uint16 and not full16Bit:
                original /= 256
                original = original.astype(np.uint8)
        layers.append(original)
    if withDir:
        return layers, opendirectory
    return layers

def getFiles(opendirectory):
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
                                             ' files does not contain exactly 1 number in the filename, which'
                                             ' is necessary for indexing Tifs by their z-layer. '
                                             'Aborting Save...'))
                error.exec_()
                return False, False
            fileIndices.append(int(numbers[0]))
    if len(files) == 0:
        error = QtGui.QMessageBox()
        error.setText(QtCore.QString('Error! There are not tif files in this folder.'))
        error.exec_()
        return False, False
    files = [y for (x, y) in sorted(zip(fileIndices, files))]
    return files, opendirectory


def fullGPUMask(cropped, side, validityMap, fullMap):
    width, height = fullMap.shape[1], fullMap.shape[0]
    cropped = cropped.reshape((width * height), 3)
    height, width, channels = fullMap.shape
    validityMap = af.data.flat(validityMap)
    xyv = fullMap.reshape(height * width, 3)
    x, y, v = np.split(xyv, 3, axis=1)
    x, y, v = af.interop.np_to_af_array(x), af.interop.np_to_af_array(y), af.interop.np_to_af_array(v)
    cropped = af.interop.np_to_af_array(cropped)
    indices = af.data.range(width*height)
    for i in af.ParallelRange(height * width):  # validitymap is v,x,y
        cx, cy, cv = x[i], y[i], v[i]
        ci = cv + cx * 256 + cy * side * 256
        color = validityMap[ci]
        pos = indices[i]
        cropped[pos, 0] *= color
        cropped[pos, 1] *= color
        cropped[pos, 2] *= color
    cropped = np.array(cropped)  # takes about 100 ms, or 2/3 of the time to do this
    cropped = cropped.reshape(height, width, 3)
    return cropped


def rgbCorrection(img, bounds, gpuMode, include, both=False):  # both refers to 8bit and 16bit returns, 8 is default
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
        else:
            undermid -= i
            undermid *= (float(predictedMid - i) / (m - i))
        undermid *= undermidmask
        if gpuMode:
            overmid = overmid.reshape(height, width)
            undermid = undermid.reshape(height, width)
        color = undermid + overmid
        img[:, :, c] = color
    full16img = img.astype(np.uint16)
    img /= 256
    img = img.astype(np.uint8)
    if both:
        return img, full16img
    return img


def rgb2xyv(rgb, radius, colorMode, only='Python'):
    '''
    :param rgb: 8bit
    :param radius:
    :param colorMode:
    :param only:
    :return:
    '''
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)
    # hsv = [0-255, 0-255, 0-255]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h = h.astype(np.float32)
    h *= 0.024639942381096416  # convert h from 255 to radians
    if colorMode == 'hsvI':
        radius *= 0.8
    elif colorMode == 'hsv':
        s = 255 - s
    s = s.astype(np.float32)
    s /= 255
    x = (s * np.cos(h) + 1) * radius  # set x
    y = (1 - s * np.sin(h)) * radius  # set y
    realSideMax = radius * 2 - 1
    x[x > realSideMax] = realSideMax
    y[y > realSideMax] = realSideMax
    xyvNumpy = np.stack((x, y, v), axis=2)
    xyvNumpy = xyvNumpy.astype(np.uint16)
    if only == 'Numpy':
        return xyvNumpy
    xyv = xyvNumpy.tolist()
    if only == 'Python':
        return xyv
    return xyv, xyvNumpy

def xyv2rgb(xyv, radius, colorMode):
    '''
    :param xyv: either a python list of three numpy x, y, v arrays as separate
    channels or a single numpy array representing an image in xyv format
    :param radius: int: defines the radius of colorspaceview
    :param colorMode: whether xyvLst is represents hsv or hsvI colorspace
    :return: rgb: 8uint numpy array: xyv image converted to rgb format
    '''
    if type(xyv) is list:
        x, y, v = xyv
    else:
        if xyv.dtype != np.float32:
            xyv = xyv.astype(np.float32)
        [x, y, v] = np.split(xyv, 3, axis=2)
    if colorMode == 'hsvI':
        dx = 1 - x / radius
        dy = y / radius - 1
        distancesqrd = np.square(dx) + np.square(dy)
        distancesqrd *= 1.25  # buffer
        s = np.sqrt(distancesqrd)
        s[s > 1] = 0
    elif colorMode == 'hsv':
        dx = 1 - x / radius
        dy = y / radius - 1
        distancesqrd = np.square(dx) + np.square(dy)
        s = 1 - np.sqrt(distancesqrd)
        s[s < 0] = 0
    h = ((np.arctan2(dy, dx) / np.pi) + 1) * 180
    hsv = np.stack((h, s, v), axis=2)  # h = [0-360], s = [0-1], v = [0-1]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb *= 255
    rgb = rgb.astype(np.uint8)
    return rgb

def xyvLst2rgb(xyvLst, radius, colorMode):
    '''
    :param xyvLst: list of [x, y, v] colors
    :param radius: int: defines the radius of colorspaceview
    :param colorMode: whether xyvLst is represents hsv or hsvI colorspace
    :return: rgbLst: lst of [r, g, b] color from [0-256]
    '''
    rgbLst = []
    for [x, y, v] in xyvLst:
        x, y = float(x), float(y)
        if colorMode == 'hsvI':
            dx = 1 - x / radius
            dy = y / radius - 1
            distancesqrd = dx ** 2 + dy ** 2
            distancesqrd *= 1.25  # buffer
            s = math.sqrt(distancesqrd)
            if s > 1:
                s = 0
        elif colorMode == 'hsv':
            dx = 1 - x / radius
            dy = y / radius - 1
            distancesqrd = dx ** 2 + dy ** 2
            s = 1 - math.sqrt(distancesqrd)
            if s < 0:
                s = 0
        h = math.atan2(dy, dx) + math.pi
        # hsv color, h:[0, 2pi], s:[0, 1], v:[0, 255]
        rgb = hsv2rgb([h, s, v])
        rgbLst.append(list(rgb))
    return rgbLst

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
    delta = float(v - min(rgb))
    s = delta / v
    if delta == 0:
        return [0, s, int(v * 255)]
    if v == r:
        h = 60 * (((g - b) / delta) % 6)
    elif v == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    return [int(h), s, int(v * 255)]

def rgbtohsv8bit(rgb):
    '''
    :param rgb: list of 3 ints: rgb color, r:[0, 255], g:[0, 255], g:[0, 255]
    :return: hsv: list of 3 ints: hsv color, h:[0, 255], s:[0, 255], v:[0, 255]
    '''
    [r, g, b] = rgb
    v = max(rgb)
    if v == 0:
        return [0, 0, 0]
    delta = float(v - min(rgb))
    s = (delta / v) * 255
    if delta == 0:
        return [0, s, int(v * 255)]
    if v == r:
        h = 42.5 * (((g - b) / delta) % 6)
    elif v == g:
        h = 42.5 * (((b - r) / delta) + 2)
    else:
        h = 42.5 * (((r - g) / delta) + 4)
    return [int(h), int(s), int(v)]

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

def getRGBMap(after, before):
    rgbMap = np.zeros((256, 256, 256), dtype=bool)
    width, height = after.shape[1], after.shape[0]
    for y in xrange(0, height):
        for x in xrange(0, width):
            if after[y, x].any() != 0:
                [r, g, b] = before[y, x]
                rgbMap[r, g, b] = True
    return rgbMap
    # this will return a RGB validityMap
