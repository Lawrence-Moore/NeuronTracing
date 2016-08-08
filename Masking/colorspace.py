from PyQt4 import QtGui, QtCore
import math
import copy
import time  # remember to do time efficiency checks
import numpy as np
import cv2
import matplotlib
import saving_and_color
import plotspace
import arrayfire as af
from PIL import Image
from color_chooser import ColorChooser

# Docstring Format:
# Param ArgName: (ArgType:) Description
# Param ArgName2: (ArgType2:) Description
# Return: (ReturnType:) Description

# Comments precede what they describe unless on same line or continuing.
# Variable description often as "type: description"

class colorSpaces():
    def __init__(self, colorspace, intensitylabel, intensityslider,
                 volumeselect, drawmenu, areaselectionview, graycheck, colormode, gpumode):
        self.colorMode = colormode  # vs 'hsv'
        self.gpuMode = gpumode
        self.view = colorspace  # QGraphicsView: graphics window for color space
        self.intensityLabel = intensitylabel  # label above slider w/intensity
        self.intensitySlider = intensityslider  # slider for intensity v in HSV
        # self.side = int: side length of the colorspace graphics window
        self.createColorSpaceView()  # create the colorspace image to scale
        self.mouseHold = False  # the mouse is not currently pressed/held
        # self.csImageVal = int: current intensity v of the colorspace view
        # self.csImage = QImage: image of the current colorspace at v = 255
        # self.currentImage = QImage: current drawings on transparent background
        # self.colorIntensityMask = QImage: black image with transparency
        # to overlay self.csImage and decrease its apparent v-intensity
        self.currentArea = []  # area/positions: [pos: [x-val:int, y-val:int]],
        # a list of points drawn by user onto colorspace (nodes of loop)
        # self.validityMap = numpy array [x[v[y:bool]]]: boolean mask of color
        # space view, where x,y: position in colorspace view, v: intensity,
        # bool:True signifies the color is kept
        self.areaMode = 'manualCreate'  # str: mode of adding areas with values
        # 'manualCreate' (creating points of polygon) vs. 'manualUpdate'
        # (changing polygon's shape) vs. 'auto' vs. 'circular' vs. 'sector'
        self.boundaryColor = QtGui.qRgba(255, 255, 255, 255)  # color of loop
        # drawn in colorspace
        self.manualSelected = -1  # point/node self.currentArea that is
        # currently selected by user when drawing polygons, -1 = none
        self.volumeMenu = volumeselect  # QComboBox: dropdown to select volume
        self.areas = [False]  # areas: [area/positions: [pos: [x-val, y-val]]]
        # list of pixels in colorspace view, which are the connected drawn loop
        self.volumes = [False]  # [self.areas]: list of volumes containing
        # areas that represent it
        self.indexVolume = 0  # int: index in self.volumes of volume that is
        # being modified/viewed in colorspace view
        self.indexArea = 0  # int: index in self.areas of area is being
        # modified/viewed in colorspace view
        self.drawMenu = drawmenu  # QComboBox: dropdown to select drawing mode
        self.areaView = areaselectionview  # QGraphicsView: Immediate left of
        # the slider, with drawn triangles that point to created area
        self.areaViewPoints = []  # [Triangles[[x, y]]] contains lists of points
        # that fill triangles in areaView
        self.drawingAreas = [False]  # [self.areaMode, self.area]
        # self.validityMap = numpyarray: a 3-dimensional [x][v][y:bool] volume
        # of colorspace that represents mask with bool:True/False
        self.nodeSize = [-2, 3]  # size of nodes drawn in colorspace
        self.validityMap = False
        self.saveGray = graycheck
        self.doneDrawing = True  # frame dropping for manualUpdate

    def addDynamicView(self, update):
        '''
        :param update: function from mip module to update dynamicView with map
        :return: none: makes a method in colorspace to call update in mip module
        '''
        self.updateDynamic = update

    def drawModeChange(self, text):
        '''
        :param text: the current string self.drawMenu
        :return: none: upon change in self.drawMenu, translate label text into
        static name for mode in attribute self.areaMode
        '''
        self.deleteArea()
        print 'im changing the drawmode to', text
        if text == 'Auto':
            self.areaMode = 'auto'
        elif text == 'Manual':
            self.areaMode = 'manualCreate'
        elif text == 'Circular':
            self.areaMode = 'circular'
        elif text == 'Sector':
            self.areaMode = 'sector'

    def createBoundary(self, x, y):
        '''
        :param x: x coordinate of mouse relative to top-left of colorspace view
        :param y: y coordinate of mouse relative to top-left of colorspace view
        :return: none: adds (x, y) coordinate to currentArea, sets pixel
        in colorspace view to boundaryColor, and pushes to display
        '''
        if (self.nodeSize[1] <= x <= (self.side - self.nodeSize[1]) and self.nodeSize[1] <= y <= (self.side - self.nodeSize[1])):
            self.currentArea.append([x, y])
            self.currentImage.setPixel(x, y, self.boundaryColor)
            self.drawColorSpaceView()

    def createPoint(self, px, py, color=False):
        # draw a filled square at point (px, py) of size self.nodeSize
        if not color:
            color = self.boundaryColor
        for dx in xrange(self.nodeSize[0], self.nodeSize[1]):
            for dy in xrange(self.nodeSize[0], self.nodeSize[1]):
                self.currentImage.setPixel(px + dx, py + dy, color)

    def sectorFromMip(self, xyv):  # xyv= either [x, y, v] or [di, df] in degrees
        self.currentArea = []
        center = self.side / 2
        angleBuffer = 0.35  # in radians, 0.4 = 45 degrees #### in preferences
        circle = 2 * math.pi
        if len(xyv) == 3:
            [x, y, v] = xyv
            self.silentSliderSet(v)
            self.currentArea.append([center, center])
            x -= center
            y -= center
            angle = math.atan2(y, x)
            aI, aF = angle - angleBuffer, angle + angleBuffer
            aI, aF = aI % circle, aF % circle
            angles = [angle]
        elif len(xyv) == 2:
            [aI, aF] = xyv
            aI, aF = 255 - aF, 255 - aI  # b/c our h in hsv is flipped
            aI, aF = aI * 0.024639942381096416, aF * 0.024639942381096416  # to radians
            angles = []
            circle = 2 * math.pi
            def splitter(i, f):  # recursive degree splitter
                if f > i:
                    dAngle = f - i
                    angle = (i + f) / 2
                else:
                    dAngle = (f + circle) - i
                    angle = ((i + f + circle) / 2) % circle
                if dAngle < angleBuffer:
                    return
                splitter(i, angle)
                angles.append(angle)
                splitter(angle, f)
            splitter(aI, aF)
        else:
            print 'error: bad/incorrect arguments were passed in! aborting'
            return
        xIUnit, yIUnit = math.cos(aI), math.sin(aI)
        xFUnit, yFUnit = math.cos(aF), math.sin(aF)
        dS = ((center - self.nodeSize[1] - 1) / 4)
        for s in xrange(dS, (center - self.nodeSize[1]), dS):
            self.currentArea.append([(int(xIUnit * s) + center), int(yIUnit * s) + center])
        for angle in angles:
            xRim, yRim = int(center + (s * math.cos(angle))), int(center + (s * math.sin(angle)))
            self.currentArea.append([xRim, yRim])
        for s in xrange(s, 1, -dS):
            self.currentArea.append([(int(xFUnit * s) + center), (int(yFUnit * s) + center)])
        self.areaMode = 'manualUpdate'  # change mode to connect nodes
        self.drawMenu.setCurrentIndex(0)
        self.manualSelected = 0
        self.updateManualBoundary(center, center, False)  # draw area with nodes in currentArea
        self.createAreaView()
        if len(xyv) != 2:
            self.createValidityMap()

    def circularFromMip(self, xyv):
        [x, y, v] = xyv
        self.intensitySlider.setValue(v)
        self.currentArea = [[x, y]]
        self.updateColorSpaceView(clear=True)
        self.createPoint(x, y)
        self.areaMode = 'circular'
        self.mouseHold = True
        self.drawColorSpaceView()

    def circularRadius(self, x, y):
        '''
        :param x: x coordinate of mouse relative to top-left of colorspace view
        :param y: y coordinate of mouse relative to top-left of colorspace view
        :return: none: draws a line from center of circle (self.currentArea[0])
        and args x,y to represent radius of circle being drawn
        '''
        if not (self.nodeSize[1] <= x <= (self.side - self.nodeSize[1]) and self.nodeSize[1] <= y <= (self.side - self.nodeSize[1])):
            return
        def recur(previous, next):
            # draw a point half-way between points [x, y] previous and next
            (ox, oy) = previous
            (nx, ny) = next
            mx = (ox + nx) / 2
            my = (oy + ny) / 2
            if (mx == nx or mx == ox) and (my == ny or my == oy):
                return  # stop dividing into segments
            recur([ox, oy], [mx, my])  # before mid-point
            self.currentImage.setPixel(mx, my, self.boundaryColor)  # mid-point
            recur([mx, my], [nx, ny])  # past mid-point
        self.updateColorSpaceView(clear=True)  # make colorspace without drawings
        [px, py] = self.currentArea[0]  # origin of circle
        recur([px, py], [x, y])  # draw line between origin and arg point x,y
        self.createPoint(px, py)  # bolden arg point x,y
        self.createPoint(x, y)  # bolden origin
        self.drawColorSpaceView()
        # get radius of circle and save it in currentArea:
        self.currentArea.append(int(math.sqrt((x - px) ** 2 + (y - py) ** 2)))

    def circularCreate(self):
        '''
        :return: none: creates a circle in colorspace view around origin and
        radius data found in currentData, where circle is polygon of 10 nodes
        '''
        self.updateColorSpaceView(clear=True)  # make colorspace without drawings
        origin = self.currentArea[0]
        radius = self.currentArea[-1]
        self.currentArea = []  # clear to contain actual nodes of polygon
        numNodes = 10                               ############# define as a constant/preferences
        radianInterval = 2 * math.pi / numNodes
        # if goes out of bounds, put in bounds
        dN = self.nodeSize[1]
        if origin[0] - radius < dN and (origin[0] - dN) < radius:
            radius = origin[0] - dN
        if origin[0] + radius > self.side - dN and (self.side - origin[0] - dN) < radius:
            radius = self.side - origin[0] - dN
        if origin[1] - radius < dN and (origin[1] - dN) < radius:
            radius = origin[1] - dN
        if origin[1] + radius > self.side - dN and (self.side - origin[1] - dN) < radius:
            radius = self.side - origin[1] - dN
        for n in xrange(0, numNodes):  # find and save location of nodes
            x = origin[0] + int(radius * math.cos(n * radianInterval))
            y = origin[1] + int(radius * math.sin(n * radianInterval))
            if not (dN <= x <= self.side - dN and dN <= y <= self.side - dN):
                self.currentArea = []
                self.displayError('Error! There was an error in '
                'bounding nodes inside the colorspace window. Please debug...')
                return
            self.currentArea.append([x, y])
        self.areaMode = 'manualUpdate'  # change mode to connect nodes
        self.drawMenu.setCurrentIndex(0)
        self.manualSelected = numNodes - 1
        self.updateManualBoundary(x, y, False) # draw area with nodes in currentArea
        self.createAreaView()

    def createManualBoundary(self, x, y):
        '''
        :param x: x coordinate of mouse relative to top-left of colorspace view
        :param y: y coordinate of mouse relative to top-left of colorspace view
        :return: none: draws a new point based one arg x,y and conencts that
        point to the previous point in currentArea. switches to manualUpdate
        mode if loop is completed.
        '''
        if (self.nodeSize[1] <= x <= (self.side - self.nodeSize[1]) and self.nodeSize[1] <= y <= (self.side - self.nodeSize[1])):
            def recur(previous, next):
                (ox, oy) = previous
                (nx, ny) = next
                mx = (ox + nx) / 2
                my = (oy + ny) / 2
                if (mx == nx or mx == ox) and (my == ny or my == oy):
                    return
                recur([ox, oy], [mx, my])  # before mid-point
                self.currentImage.setPixel(mx, my, self.boundaryColor) # discovered mid-point
                recur([mx, my], [nx, ny])  # past mid-point
            if len(self.currentArea) > 2:
                for i, [ox, oy] in enumerate(self.currentArea):
                    distance = self.nodeSize[1] * 2 - 1
                    if abs(ox - x) < distance and abs(oy - y) < distance:
                        # previous point is distance away to x,y, loop is done
                        self.manualSelected = i
                        self.updateManualBoundary(ox, oy, False)
                        self.areaMode = 'manualUpdate'
                        self.createValidityMap()
                        self.createAreaView()
                        return
            self.currentArea.append([x, y])
            self.createPoint(x, y)
            if len(self.currentArea) == 1:  # a loop of one node has no lines
                self.drawColorSpaceView()
                return
            recur(self.currentArea[-2], self.currentArea[-1])  # draw line
            # connecting new point arg x,y to last point in currentArea
            self.drawColorSpaceView()

    def updateManualBoundary(self, x, y, selecting):
        '''
        :param x: x coordinate of mouse relative to top-left of colorspace view
        :param y: y coordinate of mouse relative to top-left of colorspace view
        :param selecting: bool: if x,y is a mouse click (not mouse drag)
        :return: none: changes manualSelected to current point to be edited if
        selecting or changes location of manualSelect point to new location x,y
        '''
        if not (self.nodeSize[1] <= x <= (self.side - self.nodeSize[1]) and self.nodeSize[1] <= y <= (self.side - self.nodeSize[1])):
            return False
        if selecting:  # if it is within (2*sizeNode - 1) pixels, select point
            distance = self.nodeSize[1] * 2 - 1
            self.manualSelected = -1
            for i, [ox, oy] in enumerate(self.currentArea):
                    if abs(ox - x) < distance and abs(oy - y) < distance:
                        self.manualSelected = i
                        self.updateManualBoundary(ox, oy, False)
            return False
        if self.manualSelected != -1:  # a point has been selected already
            # this differs from previous 'recur' b/c it appends to newarea
            def recur(previous, next):
                (ox, oy) = previous
                (nx, ny) = next
                mx = (ox + nx) / 2
                my = (oy + ny) / 2
                if (mx == nx or mx == ox) and (my == ny or my == oy):
                    return
                recur([ox, oy], [mx, my])  # before mid-point
                self.currentImage.setPixel(mx, my, self.boundaryColor)  # discovered mid-point
                newarea.append([mx, my])
                recur([mx, my], [nx, ny])  # past mid-point
            self.updateColorSpaceView(clear=True)  # remove drawings from view
            self.currentArea[self.manualSelected] = [x, y]  # move to mouse
            # resave area and drawing information
            self.drawingAreas[self.indexArea] = ['manual']
            self.packCurrentArea2DrawingAreas()
            # redraw points and lines in polygon
            previous = self.currentArea[0]
            newarea = [previous]
            self.createPoint(previous[0], previous[1])
            for next in self.currentArea[1::]:
                self.createPoint(next[0], next[1])
                recur(previous, next)
                newarea.append(next)
                previous = next
            recur(previous, self.currentArea[0])  # complete the loop
            # the currently selected point is red in color
            self.createPoint(x, y, color=QtGui.qRgba(255, 100, 100, 255))
            self.drawColorSpaceView()
            self.boundaryToAreas(newarea)

    def boundaryToAreas(self, newarea):
        '''
        :param newarea: area of [x, y] points that consist of drawn loop
        :return: none: separates newarea into x and y lists for efficient
        mapping in createValidityMap and save in self.areas, creates new
        mapping with area and pushes to colorspace view
        '''
        newarea = np.array(newarea, dtype=np.float32)
        newarea /= self.side
        areaX, areaY = newarea[:, 0], newarea[:, 1]
        self.areas[self.indexArea] = [areaX, areaY, copy.copy(self.csImageVal)]

    def packCurrentArea2DrawingAreas(self):
        drawingArea = np.array(self.currentArea, dtype=np.float32)
        drawingArea /= self.side
        self.drawingAreas[self.indexArea].append(drawingArea)

    def finalizeBoundary(self):
        '''
        :return: none: after drawing points for loop in 'auto' mode, draws its
        connections and finds its intersection, cropping to only the closed part
        '''
        if len(self.currentArea) == 0:  # shouldn't be called in this case
            return  # but it fixed somethings
        self.drawingAreas[self.indexArea] = ['auto']
        self.packCurrentArea2DrawingAreas()
        newarea = []
        self.intersection = False  # there is an intersection in drawn polygon
        def drawlines():
            def existsIntersection(x, y):  # the point is surround by others
                if len(newarea) < 5:
                    return False
                if [x, y] in newarea:  # 3 mid-point intersection cases:
                    self.intersection = [x, y]
                    return True
                elif [x + 1, y] in newarea and [x, y + 1] in newarea:
                    newarea.append([x, y])
                    self.intersection = [x + 1, y]
                    return True
                elif [x - 1, y] in newarea and [x, y - 1] in newarea:  # last condition may be redundant
                    newarea.append([x, y])
                    self.intersection = [x - 1, y]
                    return True
            [pox, poy] = self.currentArea[0]
            for [px, py] in self.currentArea[1::]:
                if [px, py] == [pox, poy]:
                    continue  # mouse mistakenly gives duplicates sometimes
                # in this version of 'recur', the point isn't drawn, but saved
                def recur(ox, oy, x, y):
                    mx = (ox + x) / 2
                    my = (oy + y) / 2
                    if (mx == x or mx == ox) and (my == y or my == oy):
                        return False
                    if recur(ox, oy, mx, my):  # before mid-point
                        return True
                    if existsIntersection(mx, my):
                        return True
                    newarea.append([mx, my])  # append discovered mid-point
                    if recur(mx, my, x, y):  # past mid-point
                        return True
                    return False
                if existsIntersection(pox, poy):
                    return True  # stop adding points when intersection is found
                newarea.append([pox, poy])  # append known point
                if recur(pox, poy, px, py):
                    return
                [pox, poy] = [px, py]
        drawlines()
        if not self.intersection:
            error = QtGui.QMessageBox()
            error.setText(QtCore.QString('Error! Could not find intersection'
                    '...this needs to debugged. Please save the traceback.'))
            error.exec_()
            self.updateColorSpaceView(clear=True)
            self.currentArea = []
            return False
        newarea = newarea[newarea.index(self.intersection)::]  # crop out start
        # update view with drawn lines
        self.updateColorSpaceView(clear=True)
        for [x, y] in newarea:
            self.currentImage.setPixel(x, y, self.boundaryColor)
        self.drawColorSpaceView()
        self.refreshAreaSpace()
        # add to self.areas and update dynamicview:
        self.boundaryToAreas(newarea)

    def updateColorSpaceView(self, redraw=False, clear=False):  # forced = forced validitymap
        '''
        :param redraw: bool: if the currentArea should be redrawn and/or pushed
        to the view
        :return: none: saves new position of intensitySlider and uses to reset
        boundaryColor. optionally redraws area and pushes to view.
        '''
        # save new intensity into csImageVal
        csImageVal = self.intensitySlider.sliderPosition()
        self.intensityLabel.setText(  # refresh intensity label with new value
            QtCore.QString(('Intensity: %d' % self.csImageVal)))
        if csImageVal > 200 and self.csImageVal <= 200:  # if light, boundaryColor is black
            self.boundaryColor = QtGui.qRgba(0, 0, 0, 255)
            if type(redraw) is not bool:
               redraw = False
        elif csImageVal < 200 and self.csImageVal >= 200:  # else if dark, boundaryColor is white
            self.boundaryColor = QtGui.qRgba(255, 255, 255, 255)
            if type(redraw) is not bool:
                redraw = False
        self.csImageVal = csImageVal
        if clear:
            self.currentImage.fill(QtGui.qRgba(0, 0, 0, 0))
        elif not (len(self.areas) == 1 or (len(self.areas) == 2 and not self.areas[1])):
            redraw = True
        if self.areas[self.indexArea]:
            self.areas[self.indexArea][2] = copy.copy(self.csImageVal)
        if type(redraw) is bool and not clear:  # redraw drawing areas
            # create transparent mask for drawing new areas on
            self.currentImage.fill(QtGui.qRgba(0, 0, 0, 0))
            if self.areas[self.indexArea]:
                # change v value at current area in self.area history
                if self.areaMode == 'auto':
                    self.finalizeBoundary()
                elif self.areaMode == 'manualUpdate':
                    [x, y] = self.currentArea[self.manualSelected]
                    self.updateManualBoundary(x, y, False)
                elif self.areaMode == 'manualCreate':
                    pass  # should be clearing self.currentArea and trashing old stuff######
            if redraw:
                print 'creating new validity map'
                self.createValidityMap()
        self.drawColorSpaceView()
        self.createAreaView()

    def createColorSpaceView(self, draw=True):
        '''
        :return: none: creates an image of a colorspace (at v = 255) that is
        scaled to the view's size and pushes to the view
        '''
        self.side = self.view.width()  # side length of view that is square
        self.csImage = QtGui.QImage(self.side, self.side, QtGui.QImage.Format_RGB32)
        self.csImageVal = self.intensitySlider.sliderPosition()
        self.intensityLabel.setText(QtCore.QString(('Intensity: %d' % self.csImageVal)))
        if self.colorMode == 'rgb':
            b = np.full((256, 256), self.csImageVal, dtype=np.uint8)
            r = g = np.arange(0, 256, 1, dtype=np.uint8)
            r, g = np.meshgrid(r, g)
            # the below 2 lines can be simplified into 1 with stacking
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r, g, b
        else:
            radius = self.side / 2
            # uses x,y coordinates on plane to find h,s locations in HSV cylinder
            # and then paints rgb color at x,y location from h,s,v=255
            line = np.arange(0, self.side, 1, dtype=np.float32)
            y = np.expand_dims(line, axis=1)
            y = np.repeat(y, self.side, axis=1)
            x = np.expand_dims(line, axis=0)
            x = np.repeat(x, self.side, axis=0)
            v = np.ones((self.side, self.side), dtype=np.float32)
            rgb = saving_and_color.xyv2rgb([x, y, v], radius, self.colorMode)
        self.csImage = QtGui.QImage(rgb, rgb.shape[1], rgb.shape[0],
                            rgb.shape[1] * 3, QtGui.QImage.Format_RGB888)
        if draw:
            # create transparent mask for drawing new areas on
            self.currentImage = QtGui.QImage(self.side, self.side, QtGui.QImage.Format_ARGB32)
            self.currentImage.fill(QtGui.qRgba(0, 0, 0, 0))
            self.drawColorSpaceView()  # push colorspace image to display

    def drawColorSpaceView(self):
        '''
        :return: none: adds csImage, colorIntensityMask, and currentImage layers
        to the colorspace view and displays it.
        '''
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, self.side, self.side)
        if self.colorMode == 'rgb':
            self.createColorSpaceView(draw=False)
            rgbImage = self.csImage.scaled(self.side, self.side)
            pic = QtGui.QPixmap.fromImage(rgbImage)
            scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        else:
            pic = QtGui.QPixmap.fromImage(self.csImage)  # add csImage
            scene.addItem(QtGui.QGraphicsPixmapItem(pic))
            self.createColorSpaceMask()
            scene.addItem(self.colorIntensityMask)  # add colorIntensityMask
        pic = QtGui.QPixmap.fromImage(self.currentImage)  # add currentImage
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        self.view.setScene(scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.show()

    def createColorSpaceMask(self):
        '''
        :return: none: creates a black mask for the colorspace view with
        transparency defined by the inverse of the current v-intensity
        '''
        mask = QtGui.QImage(self.side, self.side, QtGui.QImage.Format_ARGB32)
        mask.fill(QtGui.qRgba(0, 0, 0, (255 - self.csImageVal)))
        pic = QtGui.QPixmap.fromImage(mask)
        self.colorIntensityMask = QtGui.QGraphicsPixmapItem(pic)

    def addArea(self):
        if self.areaMode == 'manualUpdate':
                self.areaMode = 'manualCreate'
        self.currentArea = []
        if not self.areas[-1]:
            self.indexArea = len(self.areas) - 1
            self.createAreaView()
            return
        self.updateColorSpaceView(redraw=True)
        self.areas.append(False)
        self.drawingAreas.append(False)
        self.indexArea += 1
        self.createAreaView()

    def deleteArea(self):
        if self.areas[self.indexArea]: # delete the old areas:
            del self.areas[self.indexArea]
            del self.drawingAreas[self.indexArea]
        if len(self.areas) == 0:
            self.areas = [False]
        if len(self.drawingAreas) == 0:
            self.drawingAreas = [False]
        self.indexArea = len(self.areas) - 1 # shift index to latest area
        if self.areas[self.indexArea]: # no buffer is at the end: deleted from end
            self.addArea()
        self.drawingAreas[-1] = False
        self.currentArea = []
        self.validityMap = False
        self.updateColorSpaceView(redraw=True)
        if len(self.areas) == 1 and not self.areas[0]:
            self.updateDynamic(self.validityMap)
        else:
            self.createValidityMap()
        if self.areaMode == 'manualUpdate':
            self.areaMode = 'manualCreate'

    def silentSliderSet(self, v):
        self.intensitySlider.blockSignals(True)
        self.intensitySlider.setValue(v)
        self.intensitySlider.blockSignals(False)

    def getPreviousArea(self, px, py):
        for i, selection in enumerate(self.areaViewPoints):
            for [x, y] in selection:
                if px == x and py == y:
                    self.indexArea = i
                    self.loadPreviousArea()
                    return

    def loadPreviousArea(self):
        self.createAreaView()
        drawType = self.drawingAreas[self.indexArea][0]
        self.silentSliderSet(self.areas[self.indexArea][2])
        if drawType == 'manual':
            self.areaMode = 'manualUpdate'
            self.manualSelected = 0
        elif drawType == 'auto':
            self.areaMode = 'auto'
        self.rescaleCurrentArea()

    def changeVolume(self, text):
        self.volumes[self.indexVolume] = [copy.deepcopy(self.areas),
        copy.deepcopy(self.areaViewPoints), copy.deepcopy(self.drawingAreas),
                                          copy.copy(self.indexArea)]
        self.indexVolume = self.volumeMenu.currentIndex()
        self.cleanCurrentArea()
        if text == 'Add Color...':
            self.addVolume()
        else:
            [self.areas, self.areaViewPoints, self.drawingAreas,
                        self.indexArea] = self.volumes[self.indexVolume]
            self.refreshAreaSpace()
            if self.drawingAreas[self.indexArea]:
                self.loadPreviousArea()
            else:
                self.updateColorSpaceView(redraw=True)

    def addVolume(self):
        self.volumes.append([])
        self.volumeMenu.setItemText(self.indexVolume, QtCore.QString('%d' % (self.indexVolume + 1)))
        self.volumeMenu.addItem(QtCore.QString('Add Color...'))
        self.areas, self.areaViewPoints, self.drawingAreas = [False], [], [False]
        self.currentArea, self.indexArea, self.manualSelected = [], 0, -1
        self.refreshAreaSpace()
        self.updateColorSpaceView(redraw=True)

    def cleanCurrentArea(self):
        self.currentArea = []
        if self.areaMode == 'manualUpdate':
            self.areaMode = 'manualCreate'

    def deleteVolume(self):
        if self.volumeMenu.currentText() == 'Add Color...':
            return
        del self.volumes[self.indexVolume]
        self.cleanCurrentArea()
        self.volumeMenu.removeItem(self.indexVolume)
        for i in xrange(self.indexVolume, (self.volumeMenu.count() - 1)):
            self.volumeMenu.setItemText(i, QtCore.QString('%d' % (i + 1)))
        if self.indexVolume != 0:
            self.indexVolume -= 1  # move for appearance + not stuck at end
        self.volumeMenu.setCurrentIndex(self.indexVolume)
        if len(self.volumes) == 0:
            self.addVolume()
        else:
            [self.areas, self.areaViewPoints, self.drawingAreas, self.indexArea] = self.volumes[self.indexVolume]
        self.refreshAreaSpace()
        if self.drawingAreas[self.indexArea]:
            self.loadPreviousArea()
        else:
            self.updateColorSpaceView(redraw=True)

    def rescaleCurrentArea(self):
        if not self.drawingAreas[self.indexArea]:
            return
        self.currentArea = self.drawingAreas[self.indexArea][1]
        self.currentArea *= self.side
        self.currentArea = self.currentArea.astype(np.uint16)
        self.currentArea = self.currentArea.tolist()
        self.updateColorSpaceView(redraw=True)

    def refreshAreaSpace(self):
        self.manualSelected = 0
        self.indexArea = len(self.areas) - 1
        self.createAreaView()
        if self.areas[self.indexArea]:
            self.silentSliderSet(self.areas[self.indexArea][2])

    def createAreaView(self):
        (width, height) = (self.areaView.width(), self.areaView.height())
        img = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.qRgba(236, 236, 236, 255))
        self.areaViewPoints = []
        yshift = width / 4  # so that 255 and 0 can be easily targeted
        for i, area in enumerate(self.areas):
            if not area:
                continue
            if i == self.indexArea:
                color = QtGui.qRgba(255, 150, 150, 255)
            else:
                color = QtGui.qRgba(0, 0, 0, 255)
            mid = int(float(height - 2 * yshift) * (1. - float(area[2]) / 255.)) + yshift
            lst = []
            for x in xrange(0, width):
                dy = (width - x) / 4
                for y in xrange((mid - dy), (mid + dy)):
                    if not (0 < x < width and 0 < y < height):
                        continue
                    img.setPixel(x, y, color)
                    lst.append([x, y])
            self.areaViewPoints.append(lst)
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        pic = QtGui.QPixmap.fromImage(img)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        self.areaView.setScene(scene)
        self.areaView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.areaView.show()
        del img

    def rescaleAreaXY(self, aX, aY):
        if self.colorMode == 'rgb':
            scale = 256
        else:
            scale = self.side
        aX, aY = aX * scale, aY * scale
        aX, aY = aX.astype(np.uint16), aY.astype(np.uint16)
        return aX, aY

    def numpyAreas2Dict(self, area, concave=False):
        # sorting function for createValidityMap
        dict = {}
        minx = self.side
        maxx = 0
        areaX, areaY = area[0], area[1]
        areaX, areaY = self.rescaleAreaXY(areaX, areaY)
        areaX, areaY = areaX.tolist(), areaY.tolist()
        for (x, y) in sorted(zip(areaX, areaY)):  # sort by the x's
            if x in dict:  # append like [min, max]
                if not concave:
                    if len(dict[x]) == 2:
                        if y > dict[x][1]:
                            dict[x][1] = y
                        elif y < dict[x][0]:
                            dict[x][0] = y
                    else:
                        if y > dict[x][0]:
                            dict[x].append(y)
                        elif y < dict[x][0]:  # if y == dict[x][0], error will be thrown
                            [temp] = dict[x]
                            del dict[x]
                            dict[x] = [y, temp]
                else:
                    dict[x].append(y)
            else:
                dict[x] = [y]
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
        while len(dict[minx]) < 2:  # don't get edge of one point
            del dict[minx]
            minx += 1
            while minx not in dict:
                minx -= 1
        dict['i'] = minx
        while len(dict[maxx]) < 2:  # don't get edge
            del dict[maxx]
            maxx -= 1
            while maxx not in dict:
                maxx -= 1
        dict['f'] = maxx
        return dict

    def createValidityMap(self, push=True):  #vxy
        a = time.time()
        if push:
            self.volumes[self.indexVolume] = [copy.deepcopy(self.areas),
            copy.deepcopy(self.areaViewPoints), copy.deepcopy(self.drawingAreas),
                                          copy.copy(self.indexArea)]
        else:
            if self.volumes[self.indexVolume]:
                self.areas = self.volumes[self.indexVolume][0]
        oneAreaMode = len(self.areas) == 1 or (len(self.areas) == 2 and not self.areas[1])
        if self.colorMode == 'rgb':
            side = 256
        else:
            side = self.side
        if self.gpuMode and oneAreaMode:
            self.validityMap = af.constant(0, 256, side, side, dtype=af.Dtype.u8)
        else:
            self.validityMap = np.zeros(shape=(256, side, side), dtype=np.uint8)
        if not self.areas[0]:
            self.validityMap = False
            self.updateDynamic(self.validityMap)
            return
        if oneAreaMode:
            dict = self.numpyAreas2Dict(self.areas[0], concave=True)
            minx, maxx = dict['i'], dict['f']
            for x in xrange(minx, (maxx + 1)):
                try:
                    bounds = dict[x]
                    if len(bounds) == 2:
                        if bounds[0] != bounds[1]:
                            newbounds = bounds
                    else:
                        newbounds = [bounds[0]]
                        for i in xrange(1, len(bounds)):
                            if abs(bounds[i] - bounds[i - 1]) > 2:
                                newbounds.append(bounds[i])
                        if len(newbounds) % 2 == 1:
                            del newbounds[-1]
                except:
                    pass  # use previous values
                for i in xrange(0, len(newbounds), 2):
                    try:
                        self.validityMap[0:256, x, newbounds[i]:newbounds[i + 1]] = 1
                    except:
                        print 'ERROR! in colorspace.createValidityMap', len(newbounds), i, x, newbounds
                        quit()
            if push:
                b = time.time()
                print 'time to create simple validitymap', 1000*(b-a)
                self.updateDynamic(self.validityMap)
            return
        # complex drawings:
        ind = []  # indices in self.areas
        indvs = []  # and there corresponding v-values
        areaDicts = []  # mini, minv
        for i, area in enumerate(self.areas):  # preprocessing areas into dicts
            if area:
                ind.append(i)  # make orderedareas
                indvs.append(area[2])  # make orderedareas
                dict = self.numpyAreas2Dict(area)
                areaDicts.append(dict)
        orderedareas = [[i, v] for (v, i) in sorted(zip(indvs, ind))]
        [lowI, lowV] = orderedareas[0]
        for [highI, highV] in orderedareas[1::]:
            if highV == lowV:
                print 'an area was discounted for having the same intensity as the other'
                continue
            lowXi = areaDicts[lowI]['i']
            lowXf = areaDicts[lowI]['f']
            highXi = areaDicts[highI]['i']
            highXf = areaDicts[highI]['f']
            lowDX = lowXf - lowXi
            highDX = highXf - highXi
            for v in xrange(lowV, (highV + 1)):
                fracV = float(v - lowV) / (highV - lowV)
                ofracV = 1 - fracV
                avgXi = int(fracV * highXi + ofracV * lowXi)
                avgXf = int(fracV * highXf + ofracV * lowXf)
                avgDX = float(avgXf - avgXi)
                for avgX in xrange(avgXi, (avgXf + 1)):
                    avgFracX = (avgX - avgXi) / avgDX
                    highX = int(avgFracX * highDX) + highXi
                    lowX = int(avgFracX * lowDX) + lowXi
                    try:
                        [highYmin, highYmax] = areaDicts[highI][highX]  # get min and max
                    except:
                        pass  # use past values
                    try:
                        [lowYmin, lowYmax] = areaDicts[lowI][lowX]  # get min max
                    except:
                        pass  # use previous values
                    avgYmin = int(highYmin * fracV + lowYmin * ofracV)
                    avgYmax = int(highYmax * fracV + lowYmax * ofracV) + 1
                    self.validityMap[v, avgX, avgYmin:avgYmax] = 1
            [lowI, lowV] = [highI, highV]
        if self.gpuMode:
            self.validityMap = af.interop.np_to_af_array(self.validityMap)
        b = time.time()
        if push:
            self.updateDynamic(self.validityMap)
        print 'time to create validityMap ms:', (b-a) * 1000
        # notes about rgb:
        # map should be in [b, r, g] format
        # right now, [b, r, g] == [v, x, y] from self.createColorSpace
        # self.createValidityMap creates in [v, x, y] format

    def plotSpace(self):
        if type(self.validityMap) is bool:
            return
        if self.gpuMode:
            validityMap = np.array(self.validityMap)
            plotspace.displayValidityMap(validityMap, 8, 500) # compression, particle size (independent vars)
        else:
            plotspace.displayValidityMap(self.validityMap, 8, 120)

    def saveStack(self, boundsinclude=False, saving=True):
        if type(self.volumes[self.indexVolume]) is bool and len(self.volumes) == 1:
            return
        if saving:
            dialog = QtGui.QFileDialog()
            opendirectory = str(dialog.getExistingDirectory())
            if opendirectory == '':
                return
        originalIndex = self.indexVolume
        maps = []
        for i in xrange(0, len(self.volumes)):
            self.indexVolume = i
            self.createValidityMap(push=False)
            if type(self.validityMap) == bool:
                continue
            maps.append(self.validityMap.copy())
        self.indexVolume = originalIndex
        [self.areas, self.areaViewPoints, self.drawingAreas,
                        self.indexArea] = self.volumes[self.indexVolume]
        if saving:
            saving_and_color.applyToStack(maps, self.view.width(),
                    opendirectory, boundsinclude, self.colorMode, self.gpuMode,
                                self.saveGray.isChecked())
        else:
            return maps

    def getOriginalColorSpace(self, after, before):
        rgbMap = np.zeros((256, 256, 256), dtype=bool)
        width, height = after.shape[1], after.shape[0]
        for y in xrange(0, height):
            for x in xrange(0, width):
                if after[y, x].any() != 0:
                    [r, g, b] = before[y, x]
                    rgbMap[r, g, b] = True
        return rgbMap
        # this will return an RGB validityMap

    def rgbClusters2Chooser(self, mipImage, boundsInclude, rgbList, prefs, dir):
        # this only works in 'hsv' mode right now
        self.chooser = ColorChooser(mipImage, boundsInclude, rgbList, self.gpuMode,
                prefs, dir, self.saveGray.isChecked(), parent=self)
        self.chooser.show()
        self.chooser.exportButton.released.connect(self.chooser2xyvNodes)

    def chooser2xyvNodes(self):
        hSpans = self.chooser.getHSpans()
        for i, hSpan in enumerate(hSpans):
            if i == 0 and len(self.volumes) == 1 and type(self.areas[0]) is bool:
                self.sectorFromMip(hSpan)
                continue
            self.volumeMenu.setCurrentIndex(len(self.volumes))
            self.changeVolume('Add Color...')
            self.sectorFromMip(hSpan)
        self.createValidityMap()

    def displayError(self, str):
        error = QtGui.QMessageBox()
        error.setText(QtCore.QString(str))
        error.exec_()






