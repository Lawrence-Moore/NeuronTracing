from PyQt4 import QtGui, QtCore
import sys
import saving_and_color
import numpy as np
import time
import copy
import arrayfire as af
from PIL import Image

sys.path.append("../Correction")
import minisom
from image_normalization import k_means, self_organizing_map, display_image

class ColorChooser(QtGui.QMainWindow):  # initiated in colorspace
    def __init__(self, mipImage, boundsInclude, rgbList, parent=None):
        QtGui.QMainWindow.__init__(self)
        self.doneInitializing = False
        self.mipImage = mipImage
        self.width, self.height = 800, 400
        self.setGeometry(QtCore.QRect(200, 200, self.width, self.height))
        self.boundsInclude = boundsInclude
        self.rgbList = rgbList
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
        self.doneInitializing = True

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

    def apply2Stack(self):
        dialog = QtGui.QFileDialog()
        opendirectory = str(dialog.getExistingDirectory())
        if opendirectory == '':
            return
        saving_and_color.applyToStack([self.merges, self.modifiedRgbList, self.rgbList], self.width, opendirectory,
                                      self.boundsInclude, 'rgbClusters', False)

    def split(self):
        i = self.selectedViews.index(True)
        img = self.displaySingleRGBImage(self.modifiedRgbList[i], rtrn=True)
        k_clustered_img, k_centers = k_means(image=img, n_colors=2, threshold=True)
        self.selectedViews[i] = False
        k_centers *= 256
        print 'shape of k_centers:', k_centers.shape
        rgbList = k_centers.astype(np.uint8)
        rgbList = rgbList.tolist()
        print 'number of colors after split:', len(rgbList)
        assert len(rgbList) == 2
        ii = self.rgbList.index(self.modifiedRgbList[i])
        self.rgbList[ii] = rgbList[0]
        self.rgbList.insert(ii, rgbList[1])
        self.modifiedRgbList[i] = rgbList[0]
        self.modifiedRgbList.insert(i, rgbList[1])
        self.reset()

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
            self.colorViews[index].setFrameStyle(QtGui.QFrame.Plain)
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

    def getHSpans(self):
        # convert rgb lists to hsv lists
        hsvList = []
        rgbList = copy.copy(self.rgbList)
        for rgb in rgbList:
            hsvList.append(saving_and_color.rgbtohsv8bit(rgb))
        # remove deletions
        originals = copy.copy(self.modifiedRgbList)
        for [mc, mcs] in self.merges:
            originals += mcs
        for i, color in enumerate(rgbList):
            if color not in originals:
                print 'deleted something...'
                del rgbList[i], hsvList[i]
        # create h and s planes
        h = np.arange(0, 256, 1, dtype=np.float32)
        masks = []
        # create distance-wise masks
        for i, [a, b, c] in enumerate(hsvList):
            mask = True
            for ii, [oa, ob, oc] in enumerate(hsvList):
                if i == ii:  # don't compare with one's self
                    continue
                if a == oa:
                    print 'warning: twin hues were found! one will be deleted'
                dH = (h - a + 122) % 255 - 122
                dOH = (h - oa + 122) % 255 - 122
                dH, dOH = np.absolute(dH), np.absolute(dOH)
                mask *= dH < dOH
            masks.append(mask)
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

    def displayRGBImage(self):
        # convert rgbModifiedList to colorModifiedList
        # convert rgbMergeList to colorMergeList
        # for color in modifiedColorList:
        #   if color in color in colorMergeList, do as below
        a = time.time()
        rmasked, gmasked, bmasked = np.split(self.mipImage.copy(), 3, axis=2)
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

    def displayRGBImageGPU(self):
        # convert rgbModifiedList to colorModifiedList
        # convert rgbMergeList to colorMergeList
        # for color in modifiedColorList:
        #   if color in color in colorMergeList, do as below
        a = time.time()
        rmasked, gmasked, bmasked = np.split(self.mipImage.copy(), 3, axis=2)
        width, height, _ = self.mipImage.shape
        # mipFloat = self.mipImage.astype(np.float32)
        mipArr = af.interop.np_to_af_array(self.mipImage)
        for i, [r, g, b] in enumerate(self.rgbList):
            mask = af.constant(0, height*width, dtype=af.Dtype.b8)
            for ii in af.ParallelRange(height * width):
                pr, pg, pb= mipArr[ii, 0], mipArr[ii, 1], mipArr[ii, 2]
                dist = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                isCloser = 1
                for ii, [ar, ag, ab] in enumerate(self.rgbList):
                    isCloser *= (0 == ((ar == r) & (ag == g) & (ab == b)))  # don't compare with one's self
                    adist = (ar - pr) ** 2 + (ag - pg) ** 2 + (ab - pb) ** 2
                    isCloser *= (dist <= adist)
                mask[ii] *= isCloser
            mask = np.array(mask)
            mask = mask.reshape(height, width)
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
        stack = [rgb]
        for [mc, mcs] in self.merges:
            if mc == rgb:
                stack = mcs
                break
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
        print 'single display time', 1000*(b-a)
        img = Image.fromarray(maskedImage)
        img.show()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = ColorChooser([[40, 92, 123], [245, 123, 42], [56, 23, 213], [123, 32, 29]])
    ex.show()
    sys.exit(app.exec_())

