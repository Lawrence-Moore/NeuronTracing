from PyQt4 import QtGui, QtCore
import math
import time
import numpy as np
import saving_and_color
import cv2
import tifffile
from editimage import EditWindow
# import arrayfire as af
import copy
import sys

# Docstring Format:
# Param ArgName: (ArgType:) Description
# Param ArgName2: (ArgType2:) Description
# Return: (ReturnType:) Description

# Comments precede what they describe unless on same line and continuing.
# Variable description often as "type: description"

class mips():
    def __init__(self, fullview, dynamicview, colorspace, editbutton, gpumode, colormode):
        self.fullView = fullview  # graphics window of full Mip
        self.dynamicView = dynamicview  # graphics window of dynamic Mip
        self.colorSpace = colorspace  # graphics window of color space
        # self.fullImage = QImage of the full MIP scaled to self.fullView
        # self.dynamicImage = QImage of the full MIP scaled to self.dynamicView
        # self.mappedMip = FullMipImage[width][height] containing a mapped
        # [x, y, v] color values for color space mask
        self.filename = ''  # nothing imported, so effectively false in value
        self.croparea = False  # [xi, yi, xf, yf] = area of full Mip image to
        # be displayed in both fullView and dynamicView
        # self.validityMap = a 3-dimensional [x][v][y:bool] volume of colorspace
        # from colorspace module that represents mask with bool:True/False
        self.editButton = editbutton
        self.editButton.setVisible(False)
        self.gpuMode = gpumode
        self.selectionMask = False
        # whether neuron locations
        self.neuronLocating = False
        self.selectedNeurons = []
        self.colorMode = colormode

    def getNeuronLocation(self, mx, my, neuronSelecting=False):
        if not self.filename:
            return
        viewWidth, viewHeight = self.fullView.width(), self.fullView.height()
        fracMx, fracMy = float(mx) / viewWidth, float(my) / viewHeight
        [xi, yi, xf, yf] = self.croparea
        width, height = xf - xi, yf - yi
        px, py = int(fracMx * width + xi), int(fracMy * height + yi)
        cx, cy, cv = self.mappedNumpyMip[py][px]
        # get array of this square around it in 5x5 pixel fashion
        r = 5
        cropped = self.mappedNumpyMip[(py - r):(py + r), (px - r):(px + r)]
        x, y, v = cropped[:, :, 0], cropped[:, :, 1], cropped[:, :, 2]
        dxy = self.colorSpace.width() / 20
        x[abs(x - cx) > dxy] = cx
        y[abs(y - cy) > dxy] = cy
        v[v < 100] = cv
        avgX, avgY, avgV = np.mean(x).astype(int), np.mean(y).astype(int), np.mean(v).astype(int)
        ########
        if type(self.selectionMask) is bool:
            self.selectionMask = QtGui.QImage(self.fullView.width(), self.fullView.width(), QtGui.QImage.Format_ARGB32)  # this assumes fullView and dynamicView are congruent
            self.selectionMask.fill(QtGui.qRgba(0, 0, 0, 0))
        px, py = float(px), float(py)
        left = int((((px - r) - xi) / width) * viewWidth - 1)
        right = int((((px + r) - xi) / width) * viewWidth + 1)
        down = int((((py - r) - yi) / height) * viewHeight - 1)
        up = int((((py + r) - yi) / height) * viewHeight + 1)
        if avgV < 210:
            color = QtGui.qRgba(255, 255, 255, 255)
        else:
            color = QtGui.qRgba(0, 0, 0, 255)
        for x in xrange(left, right):
            self.selectionMask.setPixel(x, up, color)
            self.selectionMask.setPixel(x, down, color)
        for y in xrange(down, up+1):
            self.selectionMask.setPixel(right, y, color)
            self.selectionMask.setPixel(left, y, color)
        self.updateMipView(cleanMask=False)
        if neuronSelecting:
            self.selectedNeurons.append([avgX, avgY, avgV])
            print self.selectedNeurons
            return
        return [avgX, avgY, avgV]

    def importImage(self):
        '''
        :return: none: asks the user for an image file via a builtin QFileDialog
        and maps it into mappedMip and pushes the image to fullView and
        dynamicView.
        '''
        dialog = QtGui.QFileDialog()
        self.filename = str(dialog.getOpenFileName())
        if not self.filename:  # user pressed cancel
            return
        self.boundsInclude = [[[0, 127, 255], [0, 127, 255], [0, 127, 255]], [True, True, True]]
        self.croparea = False
        QtGui.QApplication.processEvents()
        self.validityMap = False
        self.createMipView()  # create the views for Mip full
        self.createMappedMip()
        self.editButton.setVisible(True) ############# delete this line

    def createMipView(self):
        '''
        :return: none: opens image from file and saves it after scaling and
        cropping as PIL Image for fullView and numpy array for dynamicView.
        pushes the image to both views as well.
        '''
        with tifffile.TIFFfile(self.filename) as tif:
            self.originalImage = tif.asarray()
        if self.originalImage.shape[2] == 4:
            self.originalImage = self.originalImage[:, :, 0:3]
        if self.originalImage.dtype == np.uint8:
            self.imageBeforeEditing = self.originalImage.astype(np.float32)
            self.imageBeforeEditing *= 256
        else:
            self.imageBeforeEditing = self.originalImage.astype(np.float32)
            self.originalImage /= 256
            self.originalImage = self.originalImage.astype(np.uint8)
        self.croparea = [0, 0, self.originalImage.shape[1], self.originalImage.shape[0]]
        self.updateMipView()

    def updateMipView(self, cleanMask=True):
        if cleanMask:
            self.selectionMask = False
        [xi, yi, xf, yf] = self.croparea
        self.fullImage = self.originalImage[yi:yf, xi:xf]
        fullResized = cv2.resize(self.fullImage, (self.fullView.width(), self.fullView.height()))
        resizedImg = QtGui.QImage(fullResized, fullResized.shape[1], fullResized.shape[0],
                            fullResized.shape[1] * 3, QtGui.QImage.Format_RGB888)
        # create scene from original image
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, self.fullView.width(), self.fullView.height())
        pic = QtGui.QPixmap.fromImage(resizedImg)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        if type(self.selectionMask) is not bool:
            img = self.selectionMask.copy()  # b/c selectionMask is destroyed after use
            pixmap = QtGui.QPixmap.fromImage(img)
            scene.addItem(QtGui.QGraphicsPixmapItem(pixmap))
        # push scene to fullView
        self.fullView.setScene(scene)
        self.fullView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.fullView.show()
        # rescale rgb data to dynamic view and save as self.dynamicImage attribute
        self.dynamicImage = cv2.resize(self.fullImage, (self.dynamicView.width(), self.dynamicView.height()))
        self.updateDynamic(self.validityMap)

    def createMappedMip(self):
        '''
        :return: none: create an [x, y, RGB] array (attribute:mappedMip) from
        the original image and convert it to [x, y, HSV] and then [x, y, XYV],
        where XYV is the coordinates of the RGB color in the graphics View in
        colorSpace module.
        '''
        a = time.time()
        radius = self.colorSpace.width() / 2
        if self.colorMode == 'rgb':
            self.mappedMip = self.originalImage.copy()
        else:
            if self.gpuMode:  # we are not ready for this (using NumpyMip) yet
                self.mappedNumpyMip = saving_and_color.rgb2xyv(self.originalImage, radius, self.colorMode, only='Numpy')
            else:
                self.mappedMip, self.mappedNumpyMip = saving_and_color.rgb2xyv(self.originalImage, radius, self.colorMode, only=False)
        b = time.time()
        print 'elapsed time for creating MappedMip ms:', (b-a)*1000

    def updateDynamic(self, map, retrn=False): #xv: [y.min, y.max]
        '''
        :param map: numpy array of dim(x, y, v, 1:bool) that represents the
        colorspace drawn. the bool is true if and only if the XYV color is to
        be drawn in the dynamicView. if not, the pixel in dynamicView is set to
        black, an RGBA of (0, 0, 0, 0).
        :return:
        '''
        if not self.neuronLocating:
            self.selectionMask = False
        if type(map) is bool:
            # if this module doesn't have its copy of the validity map and one
            # wasn't passed in (map=False), don't crop anything and redraw
            if not map or type(self.validityMap) is bool:
                self.validityMap = False
                if self.filename:
                    self.drawDynamicView(self.dynamicImage)
                return
        else:  # push in the given map into this module's map
            self.validityMap = map
        a = time.time()
        if self.gpuMode:
            cropped = saving_and_color.fullGPUMask(self.originalImage,
                self.colorSpace.width(), self.validityMap, self.mappedNumpyMip)
            if retrn:
                return cropped
            [xi, yi, xf, yf] = self.croparea
            cropped = cropped[yi:yf, xi:xf]  # takes 30 ms to crop and resize
            cropped = cv2.resize(cropped, (self.dynamicView.width(), self.dynamicView.height()))
            b = time.time()
            print 'time to mask with gpu:', (b-a)*1000
            self.drawDynamicView(cropped)
            return
        if not retrn:
            [xi, yi, xf, yf] = self.croparea  # get area to be shown in dynamicView
            # get sizes of dynamicView and cropped area
            width, height = (self.dynamicView.width(), self.dynamicView.height())
            (mwidth, mheight) = (xf - xi, yf - yi)
            fracw = float(mwidth) / width  # for every pixel px movement in the
            # dynamicView, fracw pixels are moved in the crop area
            frach = float(mheight) / height  # same from py and frach
            my = yi  # the initial y value of the crop area
            cropped = self.dynamicImage.copy()  # copy data containing cropped area
            # flatten the X-Y dims so that needed indices can be assembled into one
            # list and set to RGBA = [0, 0, 0, 0] (much quicker than individually)
        else:
            cropped = self.originalImage.copy()
            width, height = self.originalImage.shape[1], self.originalImage.shape[0]
        cropped = cropped.reshape((width * height), 3)
        indices = []
        if retrn:
            for py in xrange(0, height):  # apply validityMap (mask) to entire image
                yshift = py * width
                for px in xrange(0, width):
                    [x, y, v] = self.mappedMip[py][px]
                    if not self.validityMap[v, x, y]:
                        indices.append((yshift + px))
        else:
            for py in xrange(0, height):  # apply validityMap (mask) to image scaled to dynamicView
                yshift = py * width
                mx = xi
                for px in xrange(0, width):
                    rmx, rmy = int(round(mx)), int(round(my))
                    [x, y, v] = self.mappedMip[rmy][rmx]
                    if not self.validityMap[v, x, y]:
                        indices.append((yshift + px))
                    mx += fracw
                my += frach

        cropped[indices] = [0, 0, 0]  # set pixels to black
        cropped = cropped.reshape(height, width, 3)  # reshape back to normal
        if retrn:
            return cropped
        else:
            b = time.time()
            print 'time to mask with cpu ms:', (b - a) * 1000
            self.drawDynamicView(cropped) # push to display

    def zoomPlus(self):
        '''
        :return: none: scales the croparea to a smaller rect of originalImage
        '''
        [xi, yi, xf, yf] = self.croparea
        width = xf - xi
        height = yf - yi
        dx = int(width / 10.)  # scale area by 1/10th smaller ############## define as a constant/preferences
        dy = int(height / 10.)
        xi, xf, yi, yf = xi + dx, xf - dx, yi + dy, yf - dy  # set new corners
        if xi == xf or yi == yf:  # corners are touching, so abort w/message
            return
        self.croparea = [xi, yi, xf, yf]  # make new croppedarea official
        self.updateMipView()  # push new croparea to fullView and dynamicView

    def zoomMinus(self):
        '''
        :return: none: scales the croparea to a larger rect of originalImage
        '''
        [xi, yi, xf, yf] = self.croparea
        width = xf - xi
        height = yf - yi
        my, mx = self.originalImage.shape[0], self.originalImage.shape[1]  # width, height of the originalImage
        if (1.25 * width) >= mx:  # can't zoom out past original size
            self.croparea = [0, 0, mx, my]  # so croparea is now entire Image
            self.createMipView()  # push new croparea to mip views
            self.updateDynamic(True)  # True: no new map to given, just refresh
            return
        dx = int(width * .125) # scale area by .125 larger ############## define as a constant/preferences
        dy = int(height * .125)
        xi, xf, yi, yf = xi - dx, xf + dx, yi - dy, yf + dy
        if xi < 0:  # zoomed out past left side
            xf -= xi
            xi = 0
        elif xf > mx:  # zoomed out past right side
            xi -= xf - mx
            xf = mx
        if yi < 0:  # zoomed out past top side
            yf -= yi
            yi = 0
        elif yf > my: # zoomed out past bottom side
            yi -= yf - my
            yf = my
        self.croparea = [xi, yi, xf, yf]  # make new croppedarea official
        self.updateMipView()  # push new croparea to fullView and dynamicView

    def shift(self, dir):
        '''
        :param dir: str: shift direction = 'r':right, 'l':left, 'd':down, 'u':up
        :return: none: scales the croparea to a larger rect of originalImage
        '''
        [xi, yi, xf, yf] = self.croparea
        # move by dx, dy as fraction of current crop area
        dx, dy = (xf - xi) / 10, (yf - yi) / 10  ############# define as a constant/preferences
        width, height = self.originalImage.shape[1], self.originalImage.shape[0]
        # move in the direction given by arg 'dir'
        if dir == 'r':
            if xf + dx > width:
                xi += width - xf
                xf = width
            else:
                xi += dx
                xf += dx
        elif dir == 'l':
            if xi - dx < 0:
                xf -= xi
                xi = 0
            else:
                xf -= dx
                xi -= dx
        elif dir == 'd':
            if yf + dy > height:
                yi += height - yf
                yf = height
            else:
                yi += dy
                yf += dy
        elif dir == 'u':
            if yi - dy < 0:
                yf -= yi
                yi = 0
            else:
                yf -= dy
                yi -= dy
        self.croparea = [xi, yi, xf, yf]  # make new croppedarea official
        self.updateMipView()  # push new croparea to fullView and dynamicView

    def drawDynamicView(self, data):
        '''
        :param data: numpy array of dim(width, height, RGB) that is of the size
        of dynamicView
        :return: none: converts the param:data into an QImage and pushes it to
        the dynamicView graphics window
        '''
        # convert from numpy array to QImage
        img = QtGui.QImage(data, data.shape[1], data.shape[0],
                            data.shape[1] * 3, QtGui.QImage.Format_RGB888)
        # create the graphics scene
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, self.dynamicView.width(), self.dynamicView.height())
        pic = QtGui.QPixmap.fromImage(img)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))  # add image to scene
        if type(self.selectionMask) is not bool:
            img = self.selectionMask.copy()  # b/c selectionMask is destroyed after use
            pixmap = QtGui.QPixmap.fromImage(img)
            scene.addItem(QtGui.QGraphicsPixmapItem(pixmap))
        # push scene to dynamicView's graphics window
        self.dynamicView.setScene(scene)
        self.dynamicView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.dynamicView.show()
        del img

    def saveImage(self):
        # warning: this saves an 8-bit image!
        dialog = QtGui.QFileDialog()
        filename = str(dialog.getSaveFileName(filter=QtCore.QString('Images (*.tif)')))
        if not filename:  # no filename was created
            return
        if type(self.validityMap) is bool:
            tifffile.imsave(filename, self.originalImage)
            return
        cropped = self.updateDynamic(self.validityMap, retrn=True)
        tifffile.imsave(filename, cropped)

    def editImage(self): ######################## the button should only be made visible after importing an image
        if not self.filename:
            return
        self.eWindow = EditWindow(self.imageBeforeEditing, self.gpuMode, self.boundsInclude, parent=self)
        self.eWindow.setGeometry(QtCore.QRect(200, 100, 950, 500))
        self.eWindow.show()
        self.eWindow.saveImageButton.released.connect(self.getEditedImage)

    def getEditedImage(self):
        self.originalImage, self.boundsInclude = self.eWindow.returnEditedImage()
        self.createMappedMip()
        self.updateMipView()

    def getDynamicPoint(self, mx, my):
        if not self.filename:
            return
        viewWidth, viewHeight = self.fullView.width(), self.fullView.height()
        fracMx, fracMy = float(mx) / viewWidth, float(my) / viewHeight
        [xi, yi, xf, yf] = self.croparea
        width, height = xf - xi, yf - yi
        px, py = int(fracMx * width + xi), int(fracMy * height + yi)
        [x, y, v] = self.mappedNumpyMip[py][px]
        print 'color at this point:', [x, y, v],
        if type(self.validityMap) is not bool:
            print 'value in validitymap:', self.validityMap[v, x, y],
        print



