from PyQt4 import QtGui, QtCore
import sys
import saving_and_color
import numpy as np
import time
import copy
# import arrayfire as af
from PIL import Image
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))
from Algorithms.clustering import k_means, self_organizing_map


class ColorChooser(QtGui.QMainWindow):  # initiated in colorspace
    def __init__(self, mipImage, boundsInclude, rgbList, gpuMode, prefs, dir, grayscale, parent=None):
        QtGui.QMainWindow.__init__(self)
        self.setStyleSheet('QMainWindow {background-color: gray;}')
        self.doneInitializing = False
        self.openDirectory = dir
        self.grayScale = grayscale
        self.shadeSelected = False  # can be changed in preferences
        self.gpuMode = gpuMode
        self.mipImage = np.array(mipImage)  # a list of numpy arrays
        self.width, self.height = 800, 400
        self.setGeometry(QtCore.QRect(200, 200, self.width, self.height))
        self.boundsInclude = boundsInclude
        self.originalRgbList = rgbList
        self.rgbList, self.modifiedRgbList = copy.copy(rgbList), copy.copy(rgbList)
        self.viewSize = 100
        self.dimRatio = 0.5
        # event from bg
        self.installEventFilter(self)
        # buttons. reset
        self.resetButton = QtGui.QPushButton(self)
        self.resetButton.setText('Reset')
        self.resetButton.released.connect(self.reset)
        # delete
        self.deleteButton = QtGui.QPushButton(self)
        self.deleteButton.setText('Delete')
        self.deleteButton.released.connect(self.delete)
        # merge
        self.mergeButton = QtGui.QPushButton(self)
        self.mergeButton.setText('Merge')
        self.mergeButton.released.connect(self.merge)
        # split
        self.splitButton = QtGui.QPushButton(self)
        self.splitButton.setText('Split')
        self.splitButton.released.connect(self.split)
        # view
        self.viewMipButton = QtGui.QPushButton(self)
        self.viewMipButton.setText('View Mip')
        self.viewMipButton.released.connect(self.viewMip)
        # export hspans to colorspace
        self.exportButton = QtGui.QPushButton(self)
        self.exportButton.setText('Export to GUI')
        # apply to stack
        self.applyStackButton = QtGui.QPushButton(self)
        self.applyStackButton.setText('Apply to Stack')
        self.applyStackButton.released.connect(self.apply2Stack)
        # enlarge aps and export buttons
        for button in [self.exportButton, self.applyStackButton]:
            button.resize((1.25 * button.width()), button.height())
        # view
        self.reset()
        self.merges = []  # [merged color, [origins of merge]]
        [self.kMeansMode, twoDMode, self.thresholdVal] = prefs
        if twoDMode:
            self.numDims = 2
        else:
            self.numDims = 3
            if self.gpuMode:
                self.create3DMasksGPU()
            else:
                self.create3DMasks()
        self.doneInitializing = True

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseButtonRelease and source in self.colorViewPorts:
            self.newSelection(self.colorViewPorts.index(source))
        elif event.type() == QtCore.QEvent.MouseButtonRelease:
            self.clearSelection()
        return False

    def resizeEvent(self, event):
        if not self.doneInitializing:
            return
        self.width, self.height = event.size().width(), event.size().height()
        x, y = self.viewSize / 8, self.viewSize / 8
        for button in [self.resetButton, self.deleteButton, self.mergeButton, self.splitButton, self.viewMipButton,
                       self.exportButton, self.applyStackButton]:
            button.move(x, y)
            x += button.width() + 5
        self.viewSize = self.height / 4
        self.reset()

    def clearSelection(self):
        print 'modified:', self.modifiedRgbList
        print 'original:', self.rgbList
        while True in self.selectedViews:
            i = self.selectedViews.index(True)
            if self.shadeSelected:
                self.color2View(self.modifiedRgbList[i], self.colorViews[i])
            self.colorViews[i].setStyleSheet('.QGraphicsView {border: 1px solid gray;}')
            self.selectedViews[i] = False

    def apply2Stack(self):
        if self.numDims == 3:
            colors = []
            for rgb in self.modifiedRgbList:
                colors.append(self.displaySingle3DImage(rgb, rtrn=True))
            saving_and_color.applyToStack(colors, (self.width / 2),
                    self.openDirectory, self.boundsInclude, 'rgbClusters3D',
                                          self.gpuMode, self.grayScale)
        else:
            dialog = QtGui.QFileDialog()
            opendirectory = str(dialog.getExistingDirectory())
            if opendirectory == '':
                return
            saving_and_color.applyToStack([self.merges, self.modifiedRgbList,
                    self.rgbList], (self.width / 2), opendirectory,
                    self.boundsInclude, 'rgbClusters', self.gpuMode, self.grayScale)

    def split(self):
        # [self.kMeansMode, twoDMode, self.thresholdVal]
        i = self.selectedViews.index(True)
        if self.numDims == 2:
            img = self.displaySingleRGBImage(self.modifiedRgbList[i], rtrn=True)
            if self.kMeansMode:
                centers = k_means(image=img, n_colors=2, threshold=self.thresholdVal)
            else:
                centers = self_organizing_map(image=img, n_colors=2, threshold=self.thresholdVal)
        else:
            # initialize progress bar
            bar = QtGui.QProgressBar()
            bar.setWindowTitle(QtCore.QString('Creating stack...'))
            bar.setWindowModality(QtCore.Qt.WindowModal)
            bar.resize((self.width * 2), self.width / 20)
            bar.move(self.width, self.width)
            bar.setMaximum(3)
            bar.show()
            img = self.displaySingle3DImage(self.modifiedRgbList[i], rtrn=True)
            bar.setValue(1)
            if self.kMeansMode:
                bar.setWindowTitle(QtCore.QString('Clustering stack with k-means...'))
                QtGui.QApplication.processEvents()
                centers = k_means(images=img, n_colors=2, threshold=self.thresholdVal)
            else:
                bar.setWindowTitle(QtCore.QString('Clustering stack with som...'))
                QtGui.QApplication.processEvents()
                centers = self_organizing_map(images=img, n_colors=2, threshold=self.thresholdVal)
        self.selectedViews[i] = False
        centers *= 256
        rgbList = centers.astype(np.uint8)
        rgbList = rgbList.tolist()
        assert len(rgbList) == 2
        originals = self.merged2Originals(self.modifiedRgbList[i])
        while len(originals) > 1:
            self.rgbList.remove(originals[-1])
            del originals[-1]
        ii = self.rgbList.index(originals[0])
        self.rgbList[ii] = rgbList[0]
        self.rgbList.insert(ii, rgbList[1])
        self.modifiedRgbList[i] = rgbList[0]
        self.modifiedRgbList.insert(i, rgbList[1])
        self.reset()
        if self.numDims == 3:
            bar.setValue(2)
            bar.setWindowTitle(QtCore.QString('Masking stack with clusters...'))
            QtGui.QApplication.processEvents()
            if self.gpuMode:
                self.create3DMasksGPU()
            else:
                self.create3DMasks()

    def viewMip(self):
        if True not in self.selectedViews:
            if self.numDims == 3:
                self.display3DImage()
            elif self.gpuMode:
                self.displayRGBImageGPU()
            else:
                self.displayRGBImage()
            return
        while True in self.selectedViews:
            i = self.selectedViews.index(True)
            if self.numDims == 3:
                self.displaySingle3DImage(self.modifiedRgbList[i])
            elif self.gpuMode:
                self.displaySingleRGBImageGPU(self.modifiedRgbList[i])
            else:
                self.displaySingleRGBImage(self.modifiedRgbList[i])
            self.colorViews[i].setStyleSheet('.QGraphicsView {border: 1px solid gray;}')
            self.color2View(self.modifiedRgbList[i], self.colorViews[i])
            self.selectedViews[i] = False

    def merge(self):
        first = self.selectedViews.index(True)
        self.selectedViews[first] = False
        colors = [self.modifiedRgbList[first]]
        while True in self.selectedViews:
            i = self.selectedViews.index(True)
            colors.append(self.modifiedRgbList[i])
            self.colorViews[i].close()
            del self.colorViewPorts[i]
            del self.selectedViews[i]
            del self.colorViews[i]
            del self.modifiedRgbList[i]
        numColors = float(len(colors))
        merged = [0, 0, 0]
        for [r, g, b] in colors:
            merged[0] += int(r / numColors)
            merged[1] += int(g / numColors)
            merged[2] += int(b / numColors)
        self.modifiedRgbList[first] = merged
        self.merges.append([merged, colors])
        self.reset(hard=False)

    def delete(self):
        while True in self.selectedViews:
            i = self.selectedViews.index(True)
            del self.modifiedRgbList[i]
            self.colorViews[i].close()
            del self.selectedViews[i]
            del self.colorViews[i]
            del self.colorViewPorts[i]
        self.reset(hard=False)

    def reset(self, hard=True):
        if self.doneInitializing:
            for view in self.colorViews:
                view.close()
        if hard:
            self.modifiedRgbList = copy.copy(self.originalRgbList)
            self.rgbList = copy.copy(self.originalRgbList)
        self.colorViews = []
        self.colorViewPorts = []
        self.selectedViews = [False] * len(self.modifiedRgbList)
        initX = 0.5 * self.viewSize
        x, y = initX, 0.5 * self.viewSize
        for i, [r, g, b] in enumerate(self.modifiedRgbList):
            colorView = QtGui.QGraphicsView(self)
            colorView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            colorView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            colorView.resize(self.viewSize, self.viewSize)
            if x > (self.width - self.viewSize - initX):
                x = initX
                y += self.viewSize * 1.2
            colorView.move(x, y)
            colorView.viewport().installEventFilter(self)
            colorView.viewport().setMouseTracking(True)
            colorView.setStyleSheet('.QGraphicsView {border: 1px solid gray;}')
            colorView.show()
            self.color2View((r, g, b), colorView)
            self.colorViews.append(colorView)
            self.colorViewPorts.append(colorView.viewport())
            x += self.viewSize * 1.2

    def color2View(self, color, view):
        r, g, b = color
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, self.viewSize, self.viewSize)
        scene.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(r, g, b)))
        view.setScene(scene)
        view.setRenderHint(QtGui.QPainter.Antialiasing)

    def newSelection(self, index):
        [r, g, b] = self.modifiedRgbList[index]
        if self.selectedViews[index]:
            self.selectedViews[index] = False
            self.colorViews[index].setStyleSheet('.QGraphicsView {border: 1px solid gray;}')
        else:
            self.selectedViews[index] = True
            self.colorViews[index].setStyleSheet('.QGraphicsView {border: 3px solid white;}')
            r, g, b = int(r * self.dimRatio), int(g * self.dimRatio), int(b * self.dimRatio)
        if self.shadeSelected:
            self.color2View((r, g, b), self.colorViews[index])

    def displayRGBImage(self):
        a = time.time()
        rmasked, gmasked, bmasked = np.split(self.mipImage.copy(), 3, axis=self.numDims)
        mipFloat = self.mipImage.astype(np.float32)
        raxis, gaxis, baxis = np.split(mipFloat, 3, axis=self.numDims)
        for i, [r, g, b] in enumerate(self.rgbList):
            mask = True
            dist = ((raxis - r) ** 2 + (gaxis - g) ** 2 + (baxis - b) ** 2)
            for ii, [ar, ag, ab] in enumerate(self.rgbList):
                if ii == i:  # don't compare with one's self
                    continue
                adist = ((raxis - ar) ** 2 + (gaxis - ag) ** 2 + (baxis - ab) ** 2)
                mask *= dist <= adist
            [r, g, b] = self.original2Merged([r, g, b])
            if [r, g, b] not in self.modifiedRgbList:
                [r, g, b] = [0, 0, 0]
            rmasked[mask] = r
            gmasked[mask] = g
            bmasked[mask] = b
        maskedImage = np.dstack((rmasked, gmasked, bmasked))
        b = time.time()
        print 'full display time with cpu', 1000*(b-a)
        img = Image.fromarray(maskedImage)
        img.show()

    def displayRGBImageGPU(self):
        a = time.time()
        height, width, _ = self.mipImage.shape
        mipArr = self.mipImage.reshape(height*width, 3)
        mipArr = af.interop.np_to_af_array(mipArr)
        channels = mipArr.copy()
        rmasked, gmasked, bmasked = channels[:, 0], channels[:, 1], channels[:, 2]
        for [r, g, b] in self.rgbList:
            mask = af.constant(0, height*width, dtype=af.Dtype.b8)
            for ii in af.ParallelRange(height * width):
                pr, pg, pb = mipArr[ii, 0], mipArr[ii, 1], mipArr[ii, 2]
                dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                isCloser = 1
                for [ar, ag, ab] in self.rgbList:
                    adist = (ar - pr) ** 2 + (ag - pg) ** 2 + (ab - pb) ** 2
                    isCloser *= (dist <= adist)
                mask[ii] = isCloser
            if [r, g, b] not in self.modifiedRgbList:
                [r, g, b] = [0, 0, 0]
            else:
                [r, g, b] = self.original2Merged([r, g, b])
            rmasked[mask], gmasked[mask], bmasked[mask] = r, g, b
        rmasked, gmasked, bmasked = rmasked.__array__(), gmasked.__array__(), bmasked.__array__()
        maskedImage = np.dstack((rmasked, gmasked, bmasked))
        maskedImage = maskedImage.reshape(height, width, 3)
        b = time.time()
        print 'full display time gpu', 1000*(b-a)
        img = Image.fromarray(maskedImage)
        img.show()

    def displaySingleRGBImageGPU(self, rgb, forceColor=False, rtrn=False):
        a = time.time()
        height, width, _ = self.mipImage.shape
        mipArr = self.mipImage.reshape(height*width, 3)
        mipArr = af.interop.np_to_af_array(mipArr)
        channels = mipArr.copy()
        rmasked, gmasked, bmasked = channels[:, 0], channels[:, 1], channels[:, 2]
        stack = self.merged2Originals(rgb)
        fullMask = af.constant(0, height*width, dtype=af.Dtype.b8)
        for [r, g, b] in stack:
            mask = af.constant(0, height*width, dtype=af.Dtype.b8)
            for ii in af.ParallelRange(height * width):
                pr, pg, pb = mipArr[ii, 0], mipArr[ii, 1], mipArr[ii, 2]
                dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                isCloser = 1
                for [ar, ag, ab] in self.rgbList:
                    adist = (ar - pr) ** 2 + (ag - pg) ** 2 + (ab - pb) ** 2
                    isCloser *= (dist <= adist)
                mask[ii] = isCloser
            fullMask |= mask
        if not forceColor:
            rgb = [0, 0, 0]
            fullMask = ~fullMask
        rmasked[fullMask], gmasked[fullMask], bmasked[fullMask] = rgb
        rmasked, gmasked, bmasked = rmasked.__array__(), gmasked.__array__(), bmasked.__array__()
        maskedImage = np.dstack((rmasked, gmasked, bmasked))
        maskedImage = maskedImage.reshape(height, width, 3)
        b = time.time()
        print 'single display time gpu', 1000*(b-a)
        if rtrn:
            return maskedImage
        img = Image.fromarray(maskedImage)
        img.show()

    def displaySingleRGBImage(self, rgb, forceColor=False, rtrn=False):
        a = time.time()
        if forceColor:
            plane = np.zeros((self.mipImage.shape[0], self.mipImage.shape[1], 1), dtype=np.uint8)
            rmasked = plane.copy()
            gmasked = plane.copy()
            bmasked = plane
        else:
            maskedImage = self.mipImage.copy()
        mipFloat = self.mipImage.astype(np.float32)
        raxis, gaxis, baxis = np.split(mipFloat, 3, axis=2)
        stack = self.merged2Originals(rgb)
        fullMask = False
        for [r, g, b] in stack:
            mask = True
            dist = ((raxis - r) ** 2 + (gaxis - g) ** 2 + (baxis - b) ** 2)
            for [ar, ag, ab] in self.rgbList:
                if [ar, ag, ab] == [r, g, b]:  # don't compare with one's self
                    continue
                adist = ((raxis - ar) ** 2 + (gaxis - ag) ** 2 + (baxis - ab) ** 2)
                mask *= dist <= adist
            fullMask += mask
        if forceColor:
            rmasked[fullMask], gmasked[fullMask], bmasked[fullMask] = rgb  # color takes over its neighbors
            maskedImage = np.dstack((rmasked, gmasked, bmasked))
        else:
            fullMask = np.repeat(fullMask, 3, axis=2)
            maskedImage[~fullMask] = 0
        b = time.time()
        if rtrn:
            return maskedImage
        print 'single display time cpu', 1000*(b-a)
        img = Image.fromarray(maskedImage)
        img.show()

    def original2Merged(self, rgb):
        def search(color):  # a recursive search
            for [mc, mcs] in self.merges:
                if color in mcs:
                    color = search(mc)
                    break
            return color
        return search(rgb)

    def merged2Originals(self, rgb):
        if type(rgb[0]) is list:
            stack = rgb
        else:
            stack = [rgb]
        def search(stack):  # a recursive search
            newstack = []
            for rgb in stack:
                if rgb not in self.rgbList:
                    found = False
                    for [mc, mcs] in self.merges:
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

    def displaySingle3DImage(self, rgb, forced=False, rtrn=False):
        a = time.time()
        stack = self.merged2Originals(rgb)
        if not forced:
            rgb = [0, 0, 0]
        if self.gpuMode:
            rmasked, gmasked, bmasked = np.split(self.mipImage.copy(), 3, axis=3)
            for color in stack:
                c = self.rgbList.index(color)
                for l, layer in enumerate(self.layerMasks):
                    mask = self.layerMasks[l][c]
                    if not forced:
                        mask = ~ mask
                    rmasked[l][mask], gmasked[l][mask], bmasked[l][mask] = rgb
        else:
            rmasked, gmasked, bmasked = self.imgChannels[0].copy(), self.imgChannels[1].copy(), self.imgChannels[2].copy()
            fullMask = False
            for color in stack:
                i = self.rgbList.index(color)
                fullMask += self.threeDMasks[i]
            if not forced:
                fullMask = ~fullMask
            rmasked[fullMask], gmasked[fullMask], bmasked[fullMask] = rgb
        maskedImage = np.concatenate((rmasked, gmasked, bmasked), axis=3)
        layers = np.split(maskedImage, self.mipImage.shape[0], axis=0)
        if rtrn:
            layers = map(lambda lst: lst[0], layers)  # the first axis has length 1
            return layers
        b = time.time()
        print 'time to display single 3D Image', 1000*(b-a)
        # create mip for maskedImage
        mip = reduce(np.maximum, layers)
        mip = mip[0]
        img = Image.fromarray(mip)
        img.show()

    def display3DImage(self, rtrn=False):
        a = time.time()
        if self.gpuMode:
            rmasked, gmasked, bmasked = np.split(self.mipImage.copy(), 3, axis=3)
            for rgb in self.modifiedRgbList:
                stack = self.merged2Originals(rgb)
                print rgb, 'color was merged from originals:', stack
                for color in stack:
                    c = self.rgbList.index(color)
                    for l, layer in enumerate(self.layerMasks):
                        mask = self.layerMasks[l][c]
                        rmasked[l][mask], gmasked[l][mask], bmasked[l][mask] = rgb
        else:
            rmasked, gmasked, bmasked = self.imgChannels[0].copy(), self.imgChannels[1].copy(), \
                                        self.imgChannels[2].copy()
            for rgb in self.modifiedRgbList:
                stack = self.merged2Originals(rgb)
                print rgb, 'color was merged from originals:', stack
                fullMask = False
                for color in stack:
                    c = self.rgbList.index(color)
                    fullMask += self.threeDMasks[c]
                rmasked[fullMask], gmasked[fullMask], bmasked[fullMask] = rgb
        maskedImage = np.concatenate((rmasked, gmasked, bmasked), axis=3)
        if rtrn:
            return maskedImage
        # create mip for maskedImage
        layers = np.split(maskedImage, self.mipImage.shape[0], axis=0)
        mip = reduce(np.maximum, layers)
        mip = mip[0]
        b = time.time()
        print 'time to display 3D Image', 1000*(b-a)
        img = Image.fromarray(mip)
        img.show()

    def create3DMasksGPU(self):
        a = time.time()
        self.layerMasks = []  # self.layerMasks[layer][color] = numpy boolean array
        layers = np.split(self.mipImage, self.mipImage.shape[0], axis=0)
        for l, layer in enumerate(layers):
            layer = layer[0]
            height, width, numcolors = layer.shape
            layerArr = layer.reshape(height*width, 3)
            layerArr = af.interop.np_to_af_array(layerArr)
            colorMasks = []
            for c, [r, g, b] in enumerate(self.rgbList):
                # print 'layer', l, 'of', len(layers), 'color', c, 'of', len(self.rgbList)
                mask = af.constant(0, height * width, dtype=af.Dtype.b8)
                for ii in af.ParallelRange(height * width):
                    pr, pg, pb = layerArr[ii, 0], layerArr[ii, 1], layerArr[ii, 2]
                    dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                    isCloser = 1
                    for [ar, ag, ab] in self.rgbList:
                        adist = (ar - pr) ** 2 + (ag - pg) ** 2 + (ab - pb) ** 2
                        isCloser *= (dist <= adist)
                    mask[ii] = isCloser
                mask = mask.__array__().astype(bool).reshape(height, width)
                colorMasks.append(mask)
            self.layerMasks.append(colorMasks)
        b = time.time()
        print 'time to create 3D masks with GPU', 1000*(b-a)

    def create3DMasks(self):
        a = time.time()
        self.imgChannels = np.split(self.mipImage, 3, axis=3)
        mipFloat = self.mipImage.astype(np.float32)
        raxis, gaxis, baxis = np.split(mipFloat, 3, axis=3)
        self.threeDMasks = []
        for c, [r, g, b] in enumerate(self.rgbList):
            print 'color', c, 'of', len(self.rgbList)
            mask = True
            dist = ((raxis - r) ** 2 + (gaxis - g) ** 2 + (baxis - b) ** 2)
            for [ar, ag, ab] in self.rgbList:
                if [ar, ag, ab] == [r, g, b]:  # don't compare with one's self
                    continue
                adist = ((raxis - ar) ** 2 + (gaxis - ag) ** 2 + (baxis - ab) ** 2)
                mask *= dist <= adist
            self.threeDMasks.append(mask)
        b = time.time()
        print 'time to create 3D Masks with CPU', 1000*(b-a)

    def getHSpans(self):
        # note: this does not unionize merged colors, it simply uses their average. it does, however
        # work with deleted colors by clustering with them and not including them in the hSpans
        # convert rgb lists to hsv lists
        hsvList = []
        rgbList = copy.copy(self.modifiedRgbList)
        for rgb in self.modifiedRgbList:
            hsvList.append(saving_and_color.rgbtohsv8bit(rgb)[0])
        hsvFullList = copy.copy(hsvList)
        for rgb in self.rgbList:  # add deleted rgb values
            merged = self.original2Merged(rgb)
            if merged not in self.modifiedRgbList:
                hsvFullList.append(saving_and_color.rgbtohsv8bit(rgb)[0])
        hsvList, hsvFullList = set(hsvList), set(hsvFullList)
        # create h field
        h = np.arange(0, 256, 1, dtype=np.float32)
        masks = []
        # create distance-wise masks
        for a in hsvList:
            mask = True
            for oa in hsvFullList:
                if a == oa:
                    continue
                dH = (h - a + 122) % 255 - 122
                dOH = (h - oa + 122) % 255 - 122
                dH, dOH = np.absolute(dH), np.absolute(dOH)
                mask *= dH < dOH
            masks.append(mask)
        colorHs = []
        for mask in masks:
            shiftedright = np.append(mask[-1], mask[:-1])
            start = mask * (~ shiftedright)
            starth = np.nonzero(start)
            if len(starth[0]) == 0:
                continue
            starth = starth[0][0]
            end = shiftedright * (~ mask)
            endh = np.nonzero(end)
            if len(endh[0]) == 0:
                continue
            endh = endh[0][0]
            colorHs.append([starth, endh])
        return colorHs

    def createHSMasks(self):
        def displayChannel(msk):
            img = Image.fromarray(msk)
            img.show()

        print 'creating masks'
        # convert rgb lists to hsv lists
        hsvList = []
        rgbList = copy.copy(self.rgbList)
        for rgb in rgbList:
            hsvList.append(saving_and_color.rgbtohsv8bit(rgb))
        # create h and s planes
        line = np.arange(0, 256, 1, dtype=np.float32)
        h = np.expand_dims(line, axis=1)
        h = np.repeat(h, 256, axis=1)
        s = np.expand_dims(line, axis=0)
        s = np.repeat(s, 256, axis=0)
        masks = []
        # create distance-wise masks
        for i, [a, b, c] in enumerate(hsvList):
            mask = True
            for ii, [oa, ob, oc] in enumerate(hsvList):
                if ii == i:  # don't compare with one's self
                    continue
                dH = (h - a + 122) % 255 - 122
                dOH = (h - oa + 122) % 255 - 122
                dS, dOS = (s - b), (s - ob)
                dist = np.square(dH) + np.square(dS)
                distO = np.square(dOH) + np.square(dOS)
                mask *= dist < distO
            # delete the below when done
            for ii, [oa, ob, oc] in enumerate(hsvList):
                if ii == i:
                    continue
                mask += (h == oa) & (s == ob)
            mask *= ~ ((h == a) * (s == b))
            masks.append(mask)
        ################################# if trying to reimplement this, follow getHSpans
        # unionize merges
        protectedColors = []
        for [mc, mcs] in self.merges:
            protectedColors.append(mcs[0])
            mi = rgbList.index(mcs[0])
            for moc in mcs[1:]:
                moi = rgbList.index(moc)
                masks[mi] += masks[moi]
                del masks[moi], rgbList[moi]
                mi = rgbList.index(mcs[0])  # in case the shifting changed the indexing
        # remove deletions
        for i, color in enumerate(rgbList):
            if color not in self.modifiedRgbList and color not in protectedColors:
                print 'deleted something...'
                del rgbList[i], masks[i]
        for mask in masks:
            slate = np.zeros((256, 256), np.uint8)
            slate[mask] = 255
            displayChannel(slate)
            # convert masks to polygons with cv2.cornerHarris()
            # return (h, s) coordinates


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    rgbList = [[40, 92, 123], [245, 123, 42], [56, 23, 213], [123, 32, 29]]
    ex = ColorChooser(False, False, rgbList, False, True, False)
    ex.show()
    sys.exit(app.exec_())

