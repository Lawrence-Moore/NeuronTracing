from PyQt4 import QtGui, QtCore
import math
import copy
import time  # remember to do time efficiency checks
import numpy as np
import cv2
import matplotlib
import saving_and_color
import plotspace
from PIL import Image

# Docstring Format:
# Param ArgName: (ArgType:) Description
# Param ArgName2: (ArgType2:) Description
# Return: (ReturnType:) Description

# Comments precede what they describe unless on same line or continuing.
# Variable description often as "type: description"

class colorSpaces():
    def __init__(self, colorspace, intensitylabel, intensityslider,
                 volumeselect, drawmenu, areaselectionview, colormode):
        self.colorMode = colormode  # vs 'hsv'
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
        self.doneDrawing = True  # implementation of dropped frames in colorspace view

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
        if 0 < x < self.side and 0 < y < self.side:
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

    def sectorFromMip(self, xyv):
        self.currentArea = []
        [x, y, v] = xyv
        self.intensitySlider.setValue(v)
        center = self.side / 2
        self.currentArea.append([center, center])
        x -= center
        y -= center
        degrees = math.atan2(y, x)
        degreeBuffer = 0.35  # in radians, 0.4 = 45 degrees
        dI, dF = degrees - degreeBuffer, degrees + degreeBuffer
        xIUnit, yIUnit = math.cos(dI), math.sin(dI)
        xFUnit, yFUnit = math.cos(dF), math.sin(dF)
        dS = ((center - 4) / 4)
        for s in xrange(dS, (center - 3), dS):
            self.currentArea.append([(int(xIUnit * s) + center), int(yIUnit * s) + center])
        xRim, yRim = int(center + (s * math.cos(degrees))), int(center + (s * math.sin(degrees)))
        self.currentArea.append([xRim, yRim])
        for s in xrange(s, 1, -dS):
            self.currentArea.append([(int(xFUnit * s) + center), (int(yFUnit * s) + center)])
        self.areaMode = 'manualUpdate'  # change mode to connect nodes
        self.drawMenu.setCurrentIndex(0)
        self.manualSelected = 0
        self.updateManualBoundary(center, center, False) # draw area with nodes in currentArea
        self.createAreaView()
        self.createValidityMap()

    def circularFromMip(self, xyv):
        [x, y, v] = xyv
        self.intensitySlider.setValue(v)
        self.currentArea = [[x, y]]
        self.updateColorSpaceView(redraw=False)
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
        if not (3 < x < (self.side - 3) and 3 < y < (self.side - 3)):
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
        self.updateColorSpaceView(False)  # make colorspace without drawings
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
        self.updateColorSpaceView(False)  # make colorspace without drawings
        origin = self.currentArea[0]
        radius = self.currentArea[-1]
        self.currentArea = []  # clear to contain actual nodes of polygon
        numNodes = 10                               ############# define as a constant/preferences
        radianInterval = 2 * math.pi / numNodes
        # if goes out of bounds, put in bounds
        if origin[0] - radius <= 3 and (origin[0] - 4) < radius:
            radius = origin[0] - 4
        if origin[0] + radius >= self.side - 3 and (self.side - origin[0] - 4) < radius:
            radius = self.side - origin[0] - 4
        if origin[1] - radius <= 3 and (origin[1] - 4) < radius:
            radius = origin[1] - 4
        if origin[1] + radius >= self.side - 3 and (self.side - origin[1] - 4) < radius:
            radius = self.side - origin[1] - 4
        for n in xrange(0, numNodes):  # find and save location of nodes
            x = origin[0] + int(radius * math.cos(n * radianInterval))
            y = origin[1] + int(radius * math.sin(n * radianInterval))
            if not (3 < x < self.side - 3 and 3 < y < self.side - 3):
                self.currentArea = []
                error = QtGui.QMessageBox()
                error.setText(QtCore.QString('Error! There was an error in '
                'bounding nodes inside the colorspace window. Please debug...'))
                error.exec_()
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
        if 3 < x < self.side - 3 and 3 < y < self.side - 3:
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
        if not (3 < x < self.side - 3 and 3 < y < self.side - 3):
            return False
        if selecting:  # if it is within (2*sizeNode - 1) pixels, select point
            distance = self.nodeSize[1] * 2 - 1
            self.manualSelected = -1
            for i, [ox, oy] in enumerate(self.currentArea):
                    if abs(ox - x) < distance and abs(oy - y) < distance:
                        self.manualSelected = i
                        self.updateManualBoundary(ox, oy, False)
            return False
        elif self.manualSelected != -1:  # a point has been selected already
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
            self.updateColorSpaceView(False)  # remove drawings from view
            self.currentArea[self.manualSelected] = [x, y]  # move to mouse
            # resave area and drawing information
            self.drawingAreas[self.indexArea] = ('manual', copy.copy(self.currentArea))
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
        areaX = []
        areaY = []
        for [x, y] in newarea:
            areaX.append(x)
            areaY.append(y)
        self.areas[self.indexArea] = (areaX, areaY, copy.copy(self.csImageVal))

    def finalizeBoundary(self):
        '''
        :return: none: after drawing points for loop in 'auto' mode, draws its
        connections and finds its intersection, cropping to only the closed part
        '''
        if len(self.currentArea) == 0:  # shouldn't be called in this case
            return  # but it fixed somethings
        self.drawingAreas[self.indexArea] = ('auto', copy.copy(self.currentArea))
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
            self.updateColorSpaceView(False)
            self.currentArea = []
            return False
        newarea = newarea[newarea.index(self.intersection)::]  # crop out start
        # update view with drawn lines
        self.updateColorSpaceView(False)
        for [x, y] in newarea:
            self.currentImage.setPixel(x, y, self.boundaryColor)
        self.drawColorSpaceView()
        # add to self.areas and update dynamicview:
        self.boundaryToAreas(newarea)

    def updateColorSpaceView(self, redraw=True):
        '''
        :param redraw: bool: if the currentArea should be redrawn and/or pushed
        to the view
        :return: none: saves new position of intensitySlider and uses to reset
        boundaryColor. optionally redraws area and pushes to view.
        '''
        # save new intensity into csImageVal
        self.csImageVal = self.intensitySlider.sliderPosition()
        self.intensityLabel.setText(  # refresh intensity label with new value
            QtCore.QString(('Intensity: %d' % self.csImageVal)))
        if self.csImageVal > 125:  # if light, boundaryColor is black
            self.boundaryColor = QtGui.qRgba(0, 0, 0, 255)
        else:  # else if dark, boundaryColor is white
            self.boundaryColor = QtGui.qRgba(255, 255, 255, 255)
        # create transparent mask for drawing new areas on
        self.currentImage.fill(QtGui.qRgba(0, 0, 0, 0))
        if redraw:  # redraw drawing areas
            if self.areas[self.indexArea]:
                # change v value at current area in self.area history
                (x, y, v) = self.areas[self.indexArea]
                self.areas[self.indexArea] = (x, y, copy.copy(self.csImageVal))
                if self.areaMode == 'auto':
                    self.finalizeBoundary()
                elif self.areaMode == 'manualUpdate':
                    [x, y] = self.currentArea[self.manualSelected]
                    self.updateManualBoundary(x, y, False)
                elif self.areaMode == 'manualCreate':
                    pass  # should be clearing self.currentArea and trashing old stuff######
                if not (len(self.areas) == 1 or (len(self.areas) == 2 and not self.areas[1])):
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
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = r, g, b
        else:
            center = self.side / 2
            # uses x,y coordinates on plane to find h,s locations in HSV cylinder
            # and then paints rgb color at x,y location from h,s,v=255
            blank = np.zeros((self.side))
            y = np.zeros((self.side, self.side))
            x = np.zeros((self.side, self.side))
            for i in xrange(0, self.side):
                temp = blank.copy()
                temp.fill(i)
                y[i] = temp
            for i in xrange(0, self.side):
                blank[i] = i
            for i in xrange(0, self.side):
                x[i] = blank
            v = np.ones((self.side, self.side))
            if self.colorMode == 'hsvI':
                dx = 1 - x / center
                dy = y / center - 1
                distancesqrd = np.square(dx) + np.square(dy)
                distancesqrd *= 1.25  # buffer
                s = np.sqrt(distancesqrd)
                s[s > 1] = 0
            elif self.colorMode == 'hsv':
                dx = 1 - x / center
                dy = y / center - 1
                distancesqrd = np.square(dx) + np.square(dy)
                s = 1 - np.sqrt(distancesqrd)
                s[s < 0] = 0
            h = ((np.arctan2(dy, dx) / np.pi) + 1) * 180
            hsv = np.zeros((self.side, self.side, 3)).astype(np.float32)
            hsv[:, :, 0] = h  # [0-360]
            hsv[:, :, 1] = s  # [0-1]
            hsv[:, :, 2] = v  # [0-1]
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            rgb *= 255
            rgb = rgb.astype(np.uint8)
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
        self.updateColorSpaceView(False)
        self.drawColorSpaceView()
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
        self.updateColorSpaceView(False)
        self.drawColorSpaceView()
        self.createAreaView()
        if len(self.areas) == 1 and not self.areas[0]:
            self.updateDynamic(False)
        else:
            self.createValidityMap()
        if self.areaMode == 'manualUpdate':
            self.areaMode = 'manualCreate'

    def getPreviousArea(self, px, py):
        for i, selection in enumerate(self.areaViewPoints):
            for [x, y] in selection:
                if px == x and py == y:
                    self.indexArea = i
                    self.createAreaView()
                    (type, area) = self.drawingAreas[self.indexArea]
                    self.intensitySlider.setValue(self.areas[self.indexArea][2])
                    self.currentArea = area
                    if type == 'manual':
                        self.areaMode = 'manualUpdate'
                        self.manualSelected = 0
                        self.updateManualBoundary(area[0][0], area[0][1], False)
                    elif type == 'auto':
                        self.areaMode = 'auto'
                        self.finalizeBoundary()
                    return

    def volumeChange(self, text):
        self.addArea()
        self.volumes[self.indexVolume] = [copy.deepcopy(self.areas),
        copy.deepcopy(self.areaViewPoints), copy.deepcopy(self.drawingAreas),
                                          copy.deepcopy(self.currentArea)]
        self.indexVolume = self.volumeMenu.currentIndex()
        if text == 'Add Volume...':
            self.volumes.append([])
            self.volumeMenu.setItemText(self.indexVolume, QtCore.QString('%d' % (self.indexVolume + 1)))
            self.volumeMenu.addItem(QtCore.QString('Add Volume...'))
            self.areas, self.areaViewPoints, self.drawingAreas = [False], [], [False]
            self.currentArea, self.indexArea, self.manualSelected = [], 0, -1
        else:
            [self.areas, self.areaViewPoints, self.drawingAreas,
                        self.currentArea] = self.volumes[self.indexVolume]
        self.refreshAreaSpace()
        self.createValidityMap()

    def deleteVolume(self):
        if self.volumeMenu.currentText() == 'Add Volume...':
            return
        elif self.volumeMenu.currentIndex == 0:
            del self.volumes[0]
            return
        del self.volumes[self.indexVolume]
        self.volumeMenu.removeItem(self.indexVolume)
        for i in xrange(self.indexVolume, (self.volumeMenu.count() - 1)):
            self.volumeMenu.setItemText(i, QtCore.QString('%d' % (i + 1)))
        if self.indexVolume != 0:
            self.indexVolume -= 1  # move for appearance + not stuck at end
        self.volumeMenu.setCurrentIndex(self.indexVolume)
        [self.areas, self.areaViewPoints, self.drawingAreas, self.currentArea] = self.volumes[self.indexVolume]
        self.refreshAreaSpace()
        self.createValidityMap()

    def refreshAreaSpace(self):
        self.manualSelected = 0
        self.indexArea = len(self.areas) - 1
        self.createAreaView()
        if self.areas[self.indexArea]:
            self.intensitySlider.setValue = self.areas[2]
        self.updateColorSpaceView()

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

    def createValidityMap(self, push=True):  #vxy
        a = time.time()
        if push:
            self.volumes[self.indexVolume] = [copy.deepcopy(self.areas),
            copy.deepcopy(self.areaViewPoints), copy.deepcopy(self.drawingAreas),
                                          copy.deepcopy(self.currentArea)]
        else:
            self.areas = self.volumes[self.indexVolume][0]
        self.validityMap = [[[False for x in xrange(0, self.side)] for x in xrange(0, self.side)] for x in xrange(0, 256)]
        if not self.areas[0]:
            self.validityMap = False
            self.updateDynamic(self.validityMap)
            return
        if len(self.areas) == 1 or (len(self.areas) == 2 and not self.areas[1]):
            (areaX, areaY, maxv) = self.areas[0]
            if self.colorMode == 'rgb':
                # map should be in [b, r, g] format
                # right now, [b, r, g] == [v, x, y] from self.createColorSpace
                # self.createValidityMap creates in [v, x, y] format
                areaX, areaY = np.array(areaX, dtype=np.uint16), np.array(areaY, dtype=np.uint16)
                ratio = 256. / self.side
                areaX, areaY = np.multiply(areaX, ratio), np.multiply(areaY, ratio)
                areaX, areaY = areaX.astype(np.uint16), areaY.astype(np.uint16)
                areaX, areaY = areaX.tolist(), areaY.tolist()
            plane = [[False for x in xrange(0, self.side)] for x in xrange(0, self.side)]
            for i, x in enumerate(areaX):
                try:
                    if areaX[i + 1] == x:  # duplicate
                        continue
                except:
                    pass
                try:
                    inew = areaX[(i + 1)::].index(x) + i + 1
                except:
                    continue
                if areaY[inew] > areaY[i]:
                    plane[x][areaY[i]:areaY[inew]] = [True] * (areaY[inew] - areaY[i])
                else:
                    plane[x][areaY[inew]:areaY[i]] = [True] * (areaY[i] - areaY[inew])
            self.validityMap[0:256] = [plane] * 256
            if push:
                b = time.time()
                print 'time to create validityMap', 1000*(b-a)
                self.updateDynamic(self.validityMap)
            return
        # complex drawings:
        ind = [] # indices in self.areas
        indvs = [] # and there corresponding v-values
        areaDicts = [] # mini, minv
        for i, area in enumerate(self.areas):  # preprocessing areas into dicts
            if area:
                ind.append(i) # make orderedareas
                indvs.append(area[2]) # make orderedareas
                dict = {}
                minx = self.side
                maxx = 0
                areaX, areaY = area[0], area[1]
                if self.colorMode == 'rgb':
                    areaX, areaY = np.array(areaX, dtype=np.uint16), np.array(areaY, dtype=np.uint16)
                    ratio = 256. / self.side
                    areaX, areaY = np.multiply(areaX, ratio), np.multiply(areaY, ratio)
                    areaX, areaY = areaX.astype(np.uint16), areaY.astype(np.uint16)
                    areaX, areaY = areaX.tolist(), areaY.tolist()
                for (x, y) in sorted(zip(areaX, areaY)):  # sort by the x's
                    if x in dict:  # append like [min, max]
                        if len(dict[x]) == 2:
                            continue
                        if y > dict[x][0]:
                            dict[x].append(y)
                        else:
                            [temp] = dict[x]
                            del dict[x]
                            dict[x] = [y, temp]
                    else:
                        dict[x] = [y]
                    if x < minx:
                        minx = x
                    if x > maxx:
                        maxx = x
                while len(dict[minx]) != 2:  # don't get edge of one point
                    del dict[minx]
                    minx += 1
                dict['i'] = minx
                while len(dict[maxx]) != 2:  # don't get edge
                    del dict[maxx]
                    maxx -= 1
                dict['f'] = maxx
                areaDicts.append(dict)
        orderedareas = [[i, v] for (v, i) in sorted(zip(indvs, ind))]
        [lowI, lowV] = orderedareas[0]
        for [highI, highV] in orderedareas[1::]:
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
                    [highYmin, highYmax] = areaDicts[highI][highX] # get min and max
                    [lowYmin, lowYmax] = areaDicts[lowI][lowX] # get min max
                    avgYmin = int(highYmin * fracV + lowYmin * ofracV)
                    avgYmax = int(highYmax * fracV + lowYmax * ofracV) + 1
                    self.validityMap[v][avgX][avgYmin:avgYmax] = [True] * (avgYmax - avgYmin)
            [lowI, lowV] = [highI, highV]
        b = time.time()
        if push:
            self.updateDynamic(self.validityMap)
        print 'time to create validityMap ms:', (b-a) * 1000

    def plotSpace(self):
        if type(self.validityMap) is bool:
            return
        plotspace.displayValidityMap(self.validityMap, 8, 120)

    def saveStack(self, boundsinclude=False, saving=True):
        if saving:
            dialog = QtGui.QFileDialog()
            opendirectory = str(dialog.getExistingDirectory())
            if opendirectory == '':
                return
        '''# initiate a progress bar
        bar = QtGui.QProgressBar()
        bar.setWindowTitle(QtCore.QString('Creating Masks...'))
        bar.setWindowModality(QtCore.Qt.WindowModal)
        size = self.view.width()
        bar.resize(size, size / 20)
        bar.move(size, size)
        currentProgress = 0
        bar.setMaximum(len(self.volumes) * 2)
        bar.show()
        QtGui.QApplication.processEvents()'''
        originalIndex = self.indexVolume
        maps = []
        for i in xrange(0, len(self.volumes)):
            self.indexVolume = i
            self.createValidityMap(push=False)
            if type(self.validityMap) == bool:
                continue
            maps.append(self.validityMap.copy())
            '''currentProgress += 1
            bar.setValue(currentProgress)
            QtGui.QApplication.processEvents()
        bar.close()'''
        self.indexVolume = originalIndex
        [self.areas, self.areaViewPoints, self.drawingAreas,
                        self.currentArea] = self.volumes[self.indexVolume]
        if saving:
            saving_and_color.applyToStack(maps, self.view.width(), opendirectory, boundsinclude, self.colorMode)
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



