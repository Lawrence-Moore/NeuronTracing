from PyQt4 import QtGui, QtCore
import sys
import saving_and_color
import numpy as np
import time
import copy
import arrayfire as af
from PIL import Image


class ColorChooser(QtGui.QMainWindow):  # initiated in colorspace
    def __init__(self, mipImage, rgbList, parent=None):
        QtGui.QMainWindow.__init__(self)
        self.doneInitializing = False
        self.mipImage = mipImage
        self.width, self.height = 800, 400
        self.setGeometry(QtCore.QRect(200, 200, self.width, self.height))
        self.rgbList = rgbList
        print self.rgbList
        self.modifiedRgbList = copy.copy(self.rgbList)
        self.viewSize = 100
        self.dimRatio = 0.5
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
        # split
        self.viewMipButton = QtGui.QPushButton(self)
        self.viewMipButton.setText('View Mip')
        self.viewMipButton.released.connect(self.viewMip)
        # view
        self.reset()
        self.merges = []  # [merged color, [origins of merge]]
        self.doneInitializing = True
        # RESIGNED!!!!!:
        # self.display = DisplayClustered(self, colorMode, self.modifiedRgbList,
        # self.merges, self.colorList, img)
        # self.display.show()

    def resizeEvent(self, event):
        if not self.doneInitializing:
            return
        self.width, self.height = event.size().width(), event.size().height()
        x, y = self.viewSize / 6, self.viewSize / 6
        self.resetButton.move(x, y)
        x += self.resetButton.width() + 5
        self.deleteButton.move(x, y)
        x += self.deleteButton.width() + 5
        self.mergeButton.move(x, y)
        x += self.mergeButton.width() + 5
        self.splitButton.move(x, y)
        x += self.splitButton.width() + 5
        self.viewMipButton.move(x, y)
        self.viewSize = self.height / 4
        self.reset()

    def split(self):
        # call clustering function to split color
        # delete color from modifiedRGB, and append latest 2 colors to modifiedRGB
        # call reset
        pass

    def viewMip(self):
        if True not in self.selectedViews:
            self.displayRGBImage()
            return
        while True in self.selectedViews:
            i = self.selectedViews.index(True)
            self.displaySingleRGBImage(self.modifiedRgbList[i])
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
            merged[0] += r / numColors
            merged[1] += g / numColors
            merged[2] += b / numColors
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
            self.modifiedRgbList = copy.copy(self.rgbList)
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
            colorView.show()
            self.color2View((r, g, b), colorView)
            self.colorViews.append(colorView)
            self.colorViewPorts.append(colorView.viewport())
            x += self.viewSize * 1.2

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseButtonRelease and source in self.colorViewPorts:
            self.newSelection(self.colorViewPorts.index(source))
        return False

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
        else:
            self.selectedViews[index] = True
            r, g, b = int(r * self.dimRatio), int(g * self.dimRatio), int(b * self.dimRatio)
        self.color2View((r, g, b), self.colorViews[index])

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
                mask ^= (h == oa) & (s == ob)
            mask *= ~ ((h == a) * (s == b))
            masks.append(mask)
        # unionize merges
        protectedColors = []
        for [mc, mcs] in self.merges:
            protectedColors.append(mc)
            mi = rgbList.index(mc)
            for moc in mcs:
                moi = rgbList.index(moc)
                masks[mi] = masks[mi] ^ masks[moi]
                del masks[moi], rgbList[moi]
                mi = rgbList.index(mc)  # in case the shifting changed the indexing
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

    def displayRGBImage(self):
        # convert rgbModifiedList to colorModifiedList
        # convert rgbMergeList to colorMergeList
        # for color in modifiedColorList:
        #   if color in color in colorMergeList, do as below
        a = time.time()
        rmasked, gmasked, bmasked = np.split(self.mipImage, 3, axis=2)
        mipFloat = self.mipImage.astype(np.float32)
        raxis, gaxis, baxis = np.split(mipFloat, 3, axis=2)
        for i, [r, g, b] in enumerate(self.rgbList):
            mask = True
            dist = ((raxis - r) ** 2 + (gaxis - g) ** 2 + (baxis - b) ** 2)
            for ii, [ar, ag, ab] in enumerate(self.rgbList):
                if ii == i:  # don't compare with one's self
                    continue
                adist = ((raxis - ar) ** 2 + (gaxis - ag) ** 2 + (baxis - ab) ** 2)
                mask *= dist <= adist
            if [r, g, b] not in self.modifiedRgbList:
                [r, g, b] = [0, 0, 0]
            else:
                for [mc, mcs] in self.merges:
                    if [r, g, b] in mcs:
                        [r, g, b] = mc
                        break
            rmasked[mask] = r
            gmasked[mask] = g
            bmasked[mask] = b
        maskedImage = np.dstack((rmasked, gmasked, bmasked))
        b = time.time()
        print 'full display time', 1000*(b-a)
        img = Image.fromarray(maskedImage)
        img.show()

    def displaySingleRGBImage(self, rgb):
        a = time.time()
        plane = np.zeros((self.mipImage.shape[0], self.mipImage.shape[1], 1), dtype=np.uint8)
        rmasked = plane.copy()
        gmasked = plane.copy()
        bmasked = plane
        mipFloat = self.mipImage.astype(np.float32)
        raxis, gaxis, baxis = np.split(mipFloat, 3, axis=2)
        stack = [rgb]
        for [mc, mcs] in self.merges:
            if mc == rgb:
                stack = mcs
                break
        for [r, g, b] in stack:
            print 'my iterations', r, g, b
            mask = True
            dist = ((raxis - r) ** 2 + (gaxis - g) ** 2 + (baxis - b) ** 2)
            for [ar, ag, ab] in self.rgbList:
                print 'inside inters', ar, ag, ab
                if [ar, ag, ab] == [r, g, b]:  # don't compare with one's self
                    continue
                adist = ((raxis - ar) ** 2 + (gaxis - ag) ** 2 + (baxis - ab) ** 2)
                mask *= dist <= adist
                print np.sum(mask)
            print rgb
            rmasked[mask] = rgb[0]
            gmasked[mask] = rgb[1]
            bmasked[mask] = rgb[2]
        maskedImage = np.dstack((rmasked, gmasked, bmasked))
        b = time.time()
        print 'single display time', 1000*(b-a)
        img = Image.fromarray(maskedImage)
        img.show()

    def updateDisplay(self):
        # pass new values to self.display
        # self.display.updateImage
        pass  # RESIGNED!!!!!


class DisplayClustered(QtGui.QMainWindow):  # initiated in ColorChooser # RESIGNED!!!!!
    def __init__(self, img, parent=None):
        QtGui.QMainWindow.__init__(self)
        try:
            af.info()
            self.gpuMode = False  # try gpu mode = True later if needed
        except:
            self.gpuMode = False
        self.img = img
        self.width, self.height = 300, 300
        self.setGeometry(QtCore.QRect(200, 200, self.width, self.height))
        self.view = QtGui.QGraphicsView()
        self.view.move(0, 0)
        self.view.resize(self.width, self.height)

    def push2Display(self):
        # depending on what format, convert self.maskedImg to 8-bit numpy rgb
        # convert numpy rgb to qimage
        # scale qimage to self.view's size
        # display qimage inside self.view
        pass

    def resizeEvent(self, event):
        self.view.resize(self.width, self.height)
        self.push2Display()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = ColorChooser([[40, 92, 123], [245, 123, 42], [56, 23, 213], [123, 32, 29]])
    ex.show()
    sys.exit(app.exec_())

