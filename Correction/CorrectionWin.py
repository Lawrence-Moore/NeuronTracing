from PyQt4 import QtGui, QtCore
import sys
from correction import Ui_CorrectionWindow
from image_normalization import *
from PIL import Image, ImageQt
import numpy as np

# Docstring Format:
# Param ArgName: (ArgType:) Description
# Param ArgName2: (ArgType2:) Description
# Return: (ReturnType:) Description

# Comments precede what they describe.

class Correction(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_CorrectionWindow()
        self.ui.setupUi(self)
        self.alignMode = False  # images not ready to be aligned
        # self.ui.beforeView = graphics window of image layers pre-processing
        # self.ui.afterView = graphics window of image layers post-processing
        # get mouse clicks from afterView
        self.ui.afterView.viewport().installEventFilter(self)
        self.ui.afterView.viewport().setMouseTracking(True)
        # set the alignment square effectively to nil
        self.selectedRect = [0, 0, 0, 0]
        self.mouseHold = False  # whether mouse is currently pressed/held
        self.ui.importButton.released.connect(self.importImage)
        self.indexLayer = 0  # initial layer displayed is 0th index
        self.ui.layerSlider.setTickPosition(self.indexLayer)
        self.filename = ''  # effectively nil/false in value
        # create QPen (color and appearance) for drawing on afterView
        self.pen = QtGui.QPen()
        self.pen.setColor(QtGui.QColor(255, 255, 255))
        # connect threshold checkbox, slider, and align widgets to methods
        self.ui.thresholdMode.stateChanged.connect(self.drawAfterView)
        self.ui.layerSlider.valueChanged.connect(self.layerChanged)
        self.ui.alignButton.pressed.connect(self.alignImages)
        # align and channel select button unavailable before import of image
        self.ui.alignButton.setVisible(False)
        self.ui.channelSelectMenu.setVisible(False)
        self.ui.layerLabel.setText(QtCore.QString('Layer: %d' % self.indexLayer))
        # self.normalizedData, self.thresholdedData = numpy arrays after
        # normalizing w/o and w/ threshold
        # self.side = side length of alignment square in pixels
        # self.fullWidth, self.fullHeight = dimensions of original images
        # self.unalignedImages = [[Original[z-Layers], Normal[z-Layers][colorLayers][QImages], Thresholded[z-Layers][colorLayers][QImages]]
        # self.alignedImages = [Normal[z-Layers][QImages], Thresholded[z-Layers][QImages]
        # self.unalignedData = [Normal[z-Layers][NumpyArrs], Thresholded[z-Layers][NumpyArrs]]
        # self.alignedData = [Normal[z-Layers][NumpyArrs], Thresholded[z-Layers][NumpyArrs]]
        self.ui.channelSelectMenu.currentIndexChanged.connect(self.channelSelectionChanged)
        self.ui.saveButton.released.connect(self.saveImage)
        self.ui.saveButton.setVisible(False)

    def saveImage(self):
        # get the filename from the save dialog
        dialog = QtGui.QFileDialog()
        filename = str(dialog.getSaveFileName(filter=QtCore.QString('Images (*.tif *.png *.jpg)')))
        if not filename:  # no filename was created
            return
        qimagemode = False  # whether the image will be a qimage or nparray
        if self.ui.thresholdMode.isChecked():
            thresh = 1
        else:
            thresh = 0
        if self.alignMode:  # view is unaligned, get from self.unalignedData
            channel = self.ui.channelSelectMenu.currentIndex()
            if channel == 0:
                img = self.unalignedData[thresh][self.indexLayer]
            else:
                qimagemode = True
                img = self.unalignedImages[(thresh + 1)][self.indexLayer][channel]
        else:  # view is aligned, get from self.alignedData
            img = self.alignedData[thresh][self.indexLayer]
        if qimagemode:  # save qimage to file with qimage module
            img.save(filename)
        else:  # save nparray to tif file with scipy module
            save_image(img, filename)

    def splitChannels(self, arr):
        '''
        :param array: numpy array: an image w/shape: (width, height, 3)
        :param channel: int (0-2): a color channel (r, g, or b)
        :return: the array param isolated to the color channel param
        '''
        maskedArrays = []
        for channel in xrange(0, 3):  # for R, G, and B channels
             # set other two channels to same value
            newarr = arr.copy()
            extraChannel = (channel + 1) % 3
            newarr[:, :, extraChannel] = newarr[:, :, channel]
            extraChannel = (channel + 2) % 3
            newarr[:, :, extraChannel] = newarr[:, :, channel]
            maskedArrays.append(newarr)
        return maskedArrays  # list of arrays where R, G, B take precedence

    def channelSelectionChanged(self):
        if self.ui.channelSelectMenu.currentIndex() == 0 and self.alignedImages:
            self.alignMode = False
            self.ui.alignButton.setVisible(False)
        else:
            self.ui.alignButton.setVisible(True)
            if not self.alignedImages:
                self.alignMode = True
            else:
                self.alignMode = False
        self.drawAfterView()

    def alignImages(self):
        if not self.filename:  # nothing was imported
            return
        if self.alignMode:  # ready to be aligned
            self.alignedImages = []
            self.alignedData = []
            # x and y are intuitively switched in align_images()
            x = int(float(self.fullHeight) / (self.ui.afterView.height()) * self.selectedRect[1])
            y = int(float(self.fullWidth) / (self.ui.afterView.width()) * self.selectedRect[0])
            width = int(float(self.fullWidth) / (self.ui.afterView.width()) * self.side)
            colorlayer = self.ui.channelSelectMenu.currentIndex() - 1
            print x, y, width, colorlayer, self.indexLayer  # my args
            # make normal aligned images
            normalAligned = align_images(self.unalignedData[0], True, x, y, width, colorlayer, self.indexLayer)
            self.alignedData.append(normalAligned)
            normalAImages = []
            for datum in normalAligned:
                normalAImages.append(self.array16ToQImage(datum))
            self.alignedImages.append(normalAImages)
            # make thresholded aligned images
            thresholdedAligned = align_images(self.unalignedData[1], True, x, y, width, colorlayer, self.indexLayer)
            self.alignedData.append(thresholdedAligned)
            threshAImages = []
            for datum in thresholdedAligned:
                threshAImages.append(self.array16ToQImage(datum))
            self.alignedImages.append(threshAImages)
            # push to views and labels
            self.ui.afterLabel.setText(QtCore.QString('After Normalizing and Aligning'))
            self.drawBeforeView()
            self.drawAfterView()
            self.ui.alignButton.setText(QtCore.QString('Unalign'))
            self.alignMode = False
            self.ui.channelSelectMenu.setVisible(False)
        else:  # already aligned, so redraw unaligned images
            self.alignedImages = []
            self.ui.alignButton.setText(QtCore.QString('Align'))
            self.alignMode = True
            self.ui.channelSelectMenu.setVisible(True)
            self.drawAfterView()

    def array16ToQImage(self, datum):
        datum = datum / 256
        datum = datum.astype(np.uint8)
        img = Image.fromarray(datum)
        img = ImageQt.ImageQt(img)
        return img

    def layerChanged(self):
        '''
        :return: none: user changed value on slider to a new layer, which is
        redrawn on beforeView and afterView.
        '''
        self.indexLayer = self.ui.layerSlider.sliderPosition()
        self.ui.layerLabel.setText(QtCore.QString('Layer: %d' % self.indexLayer))
        self.drawBeforeView()
        self.drawAfterView()

    def eraseRect(self):
        '''
        :return: none: effectively the rectangle to nil and redraws afterView.
        '''
        self.selectedRect = [0, 0, 0, 0]
        self.drawAfterView()

    def drawBeforeView(self):
        '''
        :return: none: draws the original layer that is designated by the slider
        widget.
        '''
        if not self.filename:  # nothing was imported
            return
        # create scene from original image
        width, height = self.ui.beforeView.width(), self.ui.beforeView.height()
        img = self.unalignedImages[0][self.indexLayer].scaled(width, height)
        pixmap = QtGui.QPixmap.fromImage(img)
        pixitem = QtGui.QGraphicsPixmapItem(pixmap)
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.addItem(pixitem)
        # push scene to beforeView
        self.ui.beforeView.setScene(scene)
        self.ui.beforeView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.beforeView.show()

    def drawAfterView(self):
        '''
        :return: none: draws the normalized, and possibly aligned, layer that is
        designated by the slider as well as the user-drawn square if it exits.
        '''
        if not self.filename:  # nothing was imported
            return
        # create scene from normalized image
        width, height = self.ui.afterView.width(), self.ui.afterView.height()
        if self.alignMode:
            if self.ui.thresholdMode.isChecked():
                img = self.unalignedImages[2][self.indexLayer][self.ui.channelSelectMenu.currentIndex()].scaled(width, height)
            else:
                img = self.unalignedImages[1][self.indexLayer][self.ui.channelSelectMenu.currentIndex()].scaled(width, height)
        else:
            if self.ui.thresholdMode.isChecked():
                img = self.alignedImages[1][self.indexLayer].scaled(width, height)
            else:
                img = self.alignedImages[0][self.indexLayer].scaled(width, height)
        pixmap = QtGui.QPixmap.fromImage(img)
        pixitem = QtGui.QGraphicsPixmapItem(pixmap)
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.addItem(pixitem)
        # add user-drawn rect to the scene if it exists, not [0, 0, 0, 0]
        [xi, yi, xf, yf] = self.selectedRect
        if self.alignMode and not (xf < xi or yf < yi):
            # make it into a square for alignment
            if xf - xi <= yf - yi:
                self.side = yf - yi  # save the square dimensions as attribute
            else:
                self.side = xf - xi
            rect = QtCore.QRectF(xi, yi, self.side, self.side)
            scene.addRect(rect, self.pen, QtGui.QBrush())
        # push entire scene to afterView
        self.ui.afterView.setScene(scene)
        self.ui.afterView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.afterView.show()

    def importImage(self):
        '''
        :return: none: asks the user for a czi file via a builtin QFileDialog
        and uses read_czi_file to convert the list of image layers into a list
        of numpy arrays. the arrays are then normalized w/ and w/o threshold and
        the new arrays and their respective images are saved as attributes of
        Correction class. the original and normalized first layers/images are
        pushed to beforeView and afterView respectively.
        '''
        dialog = QtGui.QFileDialog()
        self.filename = str(dialog.getOpenFileName(filter=QtCore.QString('CZI File (*.czi)')))
        if not self.filename:  # user pressed cancel
            return
        # initial layer displayed is 0th index
        self.ui.layerSlider.setTickPosition(0)
        self.indexLayer = 0
        self.unalignedData = []
        self.alignedImages = []
        self.unalignedImages = []
        self.alignedData = []
        # convert czi file to numpy arrays
        originalData = read_czi_file(self.filename)
        # initiate a progress bar
        bar = QtGui.QProgressBar()
        bar.setWindowTitle(QtCore.QString('Importing Image...'))
        bar.setWindowModality(QtCore.Qt.WindowModal)
        bar.resize((self.ui.beforeView.width()), (self.ui.beforeView.width() / 20))
        bar.move(self.ui.beforeView.width(), self.ui.beforeView.width())
        bar.setMinimum(0)
        currentProgress = 0
        bar.setMaximum((len(originalData) * 12))
        bar.show()
        QtGui.QApplication.processEvents()
        # save dimensions of imported images as attributes for later
        self.fullHeight, self.fullWidth = originalData[0].shape[0], originalData[0].shape[1]
        # set number of layers in image to slide through with slider widget
        self.ui.layerSlider.setMaximum((len(originalData) - 1))
        # convert numpy arrays to QImages and save as attribute
        originalLayers = []
        for data in originalData:
            data = data / 256
            data = data.astype(np.uint8)
            img = Image.fromarray(data)
            img = ImageQt.ImageQt(img)
            originalLayers.append(img)
            # update progress bar
            currentProgress += 4
            bar.setValue(currentProgress)
            QtGui.QApplication.processEvents()
        self.unalignedImages.append(originalLayers)
        self.drawBeforeView()  # push the original image to beforeView
        # normalize original images without threshold and save as attribute
        normalizedData = normalize_colors(originalData)
        self.unalignedData.append(normalizedData)
        normalizedImages = []
        for data in normalizedData:  # for all layers after normalizing
            data = data / 256
            data = data.astype(np.uint8)  # convert to 8-bit for images
            colorData = [data]
            colorData += self.splitChannels(data)
            colorImages = []
            # make list of normalized images at this layer by color channel
            for datum in colorData:
                img = Image.fromarray(datum)
                img = ImageQt.ImageQt(img)
                colorImages.append(img)
                currentProgress += 1
                bar.setValue(currentProgress)
                QtGui.QApplication.processEvents()
            normalizedImages.append(colorImages)
            # update progress bar
        self.unalignedImages.append(normalizedImages)
        # normalize original images with threshold and save as attribute
        thresholdedData = normalize_colors(originalData, True)
        self.unalignedData.append(thresholdedData)
        thresholdedImages = []
        for data in thresholdedData:  # for all layers after thresholding
            data = data / 256
            data = data.astype(np.uint8)  # convert to 8-bit for images
            colorData = [data]
            colorData += self.splitChannels(data)
            colorImages = []
            # make list of thresholded images at this layer by color channel
            for datum in colorData:
                img = Image.fromarray(datum)
                img = ImageQt.ImageQt(img)
                colorImages.append(img)
            thresholdedImages.append(colorImages)
            currentProgress += 4
            bar.setValue(currentProgress)
            QtGui.QApplication.processEvents()
        self.unalignedImages.append(thresholdedImages)
        self.alignMode = True  # afterView is ready to draw rects with mouse
        self.drawAfterView()  # push to processed images afterView
        self.ui.afterLabel.setText(QtCore.QString('After Normalizing'))
        self.ui.channelSelectMenu.setVisible(True)  # past normalizing, so ready to align
        self.ui.saveButton.setVisible(True)

    def eventFilter(self, source, event):
        '''
        :param source: widget that created the event
        :param event: the event object
        :return: none: responds to mouse clicks in the afterView by drawing
        squares if image is imported.
        '''
        if source == self.ui.afterView.viewport() and self.alignMode:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.mouseHold = True
                pos = event.pos()
                self.selectedRect[0], self.selectedRect[1] = pos.x(), pos.y()
            elif event.type() == QtCore.QEvent.MouseMove and self.mouseHold:
                pos = event.pos()
                self.selectedRect[2], self.selectedRect[3] = pos.x(), pos.y()
                self.drawAfterView()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self.mouseHold = False
                pos = event.pos()
                self.selectedRect[2], self.selectedRect[3] = pos.x(), pos.y()
                self.drawAfterView()
        return False

    def keyPressEvent(self, event):
        '''
        :param event: the user pressed the key
        :return: none: erases drawn square on afterView after delete or 'c'
        (for clear) keys are pressed. quits after escape key is pressed.
        attempts to align/unalign images after 'a' is pressed.
        '''
        key = event.key()
        if key == QtCore.Qt.Key_Delete or key == QtCore.Qt.Key_C:
            self.eraseRect()
        elif key == QtCore.Qt.Key_Escape:
            self.close()
            quit()
        elif key == QtCore.Qt.Key_A:
            self.alignImages()

    def resizeEvent(self, event):
        '''
        :param event: the user changed the size of the main window
        :return: sets the position and size of all widgets based on the
        mainwindow's width and height.
        '''
        width, height = event.size().width(), event.size().height()
        y = .15 * height
        x = .02 * width  # margins from side of window
        side = height - 1.4 * y  # side length of views
        # adjust views:
        self.ui.beforeView.move(x, y)  # margin from left
        self.ui.beforeView.resize(side, side)
        self.ui.afterView.move((width - x - side), y)  # margin from right
        self.ui.afterView.resize(side, side)
        # adjust slider:
        self.ui.layerSlider.resize(26, side)
        self.ui.layerSlider.move(((width / 2) - (self.ui.layerSlider.width() / 2)), y)
        # adjust labels:
        y = .115 * height  # margin from top of window for descriptive labels
        self.ui.beforeLabel.resize(side, 20)
        self.ui.beforeLabel.move(x, y)
        self.ui.afterLabel.resize(side, 20)
        self.ui.afterLabel.move((width - x - side), y)
        self.ui.layerLabel.move(((width / 2) - (self.ui.layerLabel.width() / 2)), y)
        # adjust import and threshold buttons
        y = .03 * height
        self.ui.importButton.move(x, y)
        x += self.ui.importButton.width() + 30
        y += 10
        self.ui.thresholdMode.move(x, y)
        # adjust align and channelselect buttons
        x = width * .98 - self.ui.alignButton.width()
        self.ui.alignButton.move(x, y)
        x -= 5 + self.ui.channelSelectMenu.width()
        self.ui.channelSelectMenu.move(x, y)
        # adjust save button
        x = width * .98 - self.ui.saveButton.width()
        y = 2
        self.ui.saveButton.move(x, y)
        # redraw views to scale
        self.drawBeforeView()
        self.drawAfterView()

if __name__ == '__main__':
    # Executes the application and creates an instance of the Correction class.
    app = QtGui.QApplication(sys.argv)
    ex = Correction()
    ex.show()
    sys.exit(app.exec_())


