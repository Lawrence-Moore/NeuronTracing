from PyQt4 import QtGui, QtCore
import sys
from correction import Ui_CorrectionWindow
from PIL import Image, ImageQt
import numpy as np
import os
import tifffile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))
from Algorithms.image_normalization import normalize_with_standard_deviation, normalize_generic
from Algorithms.helper_functions import read_czi_file, generate_mip, save_image
from Algorithms.alignment import align_images


# Docstring Format:
# Param ArgName: (ArgType:) Description
# Param ArgName2: (ArgType2:) Description
# Return: (ReturnType:) Description

# Comments precede what they describe unless on same line or continuing.
# Variable description often as "type: description"


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
        self.selectedRect = [0, 0, 0, 0]  # [xi, yi, xf, yf]
        self.mouseHold = False  # whether mouse is currently pressed/held
        self.ui.importButton.released.connect(self.importImage)
        self.indexLayer = 0  # initial layer displayed is 0th index
        self.ui.layerSlider.setTickPosition(self.indexLayer)
        self.filename = ''  # effectively nil/false in value
        # create QPen (color and appearance) for drawing rectangle in afterView
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
        # self.side = side length of alignment square in pixels
        # self.fullWidth, self.fullHeight = dimensions of original images
        # self.unalignedImages = [[Original[z-Layers:QImages],
        # Normal[z-Layers]colorLayers:QImages], Thresholded[z-Layers][colorLayers:QImages]]
        # self.alignedImages = [Normal[z-Layers:QImages], Thresholded[z-Layers:QImages]
        # self.unalignedData = [Normal[z-Layers:NumpyArrs], Thresholded[z-Layers:NumpyArrs]]
        # self.alignedData = [Normal[z-Layers:NumpyArrs] , Thresholded[z-Layers:NumpyArrs]]
        # connect widgets to actions and set initial visibility
        self.ui.channelSelectMenu.currentIndexChanged.connect(self.channelSelectionChanged)
        self.ui.saveButton.released.connect(self.saveImage)
        self.ui.saveButton.setVisible(False)
        self.ui.saveStackButton.released.connect(self.saveStack)
        self.ui.saveStackButton.setVisible(False)
        self.ui.saveMIPButton.released.connect(self.saveMIP)
        self.ui.saveMIPButton.setVisible(False)
        self.ui.thresholdMode.setVisible(False)
        self.ui.thresholdMenu.activated[str].connect(self.changeThreshold)
        self.ui.thresholdMenu.setVisible(False)
        self.ui.alignField.setVisible(False)

    def changeThreshold(self, i):
        '''
        :param i: str: new thresholding value from self.ui.thresholdMenu
        :return: none: renormalizes original imported file with new threshold
        value and changes self's unaligned Data and Images, as well as aligned
        if necessary.
        '''
        # read in original data from imported czi
        originalData, thresholdVal = read_czi_file(self.filename), int(i)
        # initiate a progress bar
        bar = QtGui.QProgressBar()
        bar.setWindowTitle(QtCore.QString('Applying new threshold of ' + i))
        bar.setWindowModality(QtCore.Qt.WindowModal)
        bar.resize((self.ui.beforeView.width()), (self.ui.beforeView.width() / 20))
        bar.move(self.ui.beforeView.width(), self.ui.beforeView.width())
        currentProgress = 0
        bar.setMaximum((len(originalData) * 2))
        bar.show()
        QtGui.QApplication.processEvents()
        # execute new threshold
        thresholdedData = normalize_with_standard_deviation(originalData, thresholdVal)
        # update progress bar
        currentProgress += len(originalData)
        bar.setValue(currentProgress)
        QtGui.QApplication.processEvents()
        # save new thresholded data
        self.unalignedData[1] = thresholdedData
        # creates image from new thresholded data and save into self
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
            # update progress bar
            currentProgress += 1
            bar.setValue(currentProgress)
            QtGui.QApplication.processEvents()
        self.unalignedImages[2] = thresholdedImages
        bar.close()
        if not self.alignMode:  # realign if images currently are
            self.alignMode = True
            self.alignImages()  # updates of afterView
        else:
            self.drawAfterView()  # else, force update of afterView

    def saveImage(self):
        '''
        :return: none: saves image currently displayed in afterView with full
        resolution and with 16uint, if it is not a single color channel.
        '''
        # get the filename from the save dialog
        dialog = QtGui.QFileDialog()
        filename = str(dialog.getSaveFileName(filter=QtCore.QString('Images (*.tif)')))
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
            else:  # saving an single color channel
                qimagemode = True  # use 8uint image, not 16uint numpy
                img = self.unalignedImages[(thresh + 1)][self.indexLayer][channel]
        else:  # view is aligned, get from self.alignedData
            img = self.alignedData[thresh][self.indexLayer]
        if qimagemode:  # save qimage to file with qimage module
            img.save(filename)
        else:  # save nparray to tif file with scipy module
            tifffile.imsave(filename, img)

    def saveStack(self):
        '''
        :return: none: saves the stack of images shown in afterView
        (normalized and optionally aligned) as full resolution, 16uint TIFs
        '''
        # get directory in which to create stackTIFs directory and save
        # dialog = QtGui.QFileDialog()
        # opendirectory = str(dialog.getExistingDirectory())
        # if not opendirectory:  # no directory was chosen
        #    return
        # create subdirectory for saving TIFs
        # delete this!:        
        savedirectory = self.filename[:-4]  # get folder name from self.filename
        if not os.path.exists(savedirectory):
            os.makedirs(savedirectory)
        savedirectory += '/stackTIFs'
        if not os.path.exists(savedirectory):
            os.makedirs(savedirectory)
        if self.ui.thresholdMode.isChecked():
            thresh = 1
            threshName = 'yesTheshold'
        else:
            thresh = 0
            threshName = 'noThreshold'
        if self.alignMode:  # view is unaligned, get from self.unalignedData
            for i, zLayerImg in enumerate(self.unalignedData[thresh]):  # revert back
                filename = savedirectory + ('/unaligned_%s z_%d.tif' % (threshName, i))
                tifffile.imsave(filename, zLayerImg)
        else:  # view is aligned, get from self.alignedData
            for i, zLayerImg in enumerate(self.alignedData[thresh]):
                filename = savedirectory + ('/aligned_%s z_%d.tif' % (threshName, i))
                tifffile.imsave(filename, zLayerImg)

    def saveMIP(self):
        '''
        :return: none: saves a mip of the stack of images shown in afterView
        '''
        dialog = QtGui.QFileDialog()
        filename = str(dialog.getSaveFileName(filter=QtCore.QString('Images (*.tif)')))
        if not filename:  # no filename was created
            return
        if self.ui.thresholdMode.isChecked():
            thresh = 1
        else:
            thresh = 0
        if self.alignMode:  # view is unaligned, get from self.unalignedImages
            mip = generate_mip(self.unalignedData[thresh])
            save_image(mip, filename)
        else:  # view is aligned, get from self.alignedImages
            mip = generate_mip(self.alignedData[thresh])
            save_image(mip, filename)

    def splitChannels(self, arr):
        '''
        :param arr: numpy array: an image w/shape: (width, height, 3)
        :return: numpy array: param arr isolated into (r, g, b) color channels
        '''
        separatedColors = []
        for channel in xrange(0, 3):  # for R, G, and B channels
            # set other two channels to same value
            newarr = arr.copy()
            extraChannel = (channel + 1) % 3
            newarr[:, :, extraChannel] = newarr[:, :, channel]
            extraChannel = (channel + 2) % 3
            newarr[:, :, extraChannel] = newarr[:, :, channel]
            separatedColors.append(newarr)
        return separatedColors  # list of arrays where R, G, B take precedence

    def channelSelectionChanged(self):
        '''
        :return: none: changes the current color channel in afterView
        '''
        if self.ui.channelSelectMenu.currentIndex() == 0:
            self.ui.alignField.setVisible(False)
            self.ui.alignButton.setVisible(False)
        else:
            self.ui.alignButton.setVisible(True)
            self.ui.alignField.setVisible(True)
        self.drawAfterView()

    def alignImages(self):
        '''
        :return: none: aligns the normalized data with the info from self's
        selectedRect (selected area), channelSelectMenu (color channel), and
        alignField (wiggle room).
        '''
        if not self.filename or self.selectedRect == [0, 0, 0, 0]:
            return  # nothing was imported or no area was selected
        if self.alignMode:  # ready to be aligned
            # push new labels
            self.ui.alignField.setVisible(False)
            self.alignMode = False
            print 'unaligned...ready for alignment'
            self.ui.channelSelectMenu.setVisible(False)
            QtGui.QApplication.processEvents()
            # initiate a progress bar
            bar = QtGui.QProgressBar()
            bar.setWindowTitle(QtCore.QString('Aligning Images'))
            bar.setWindowModality(QtCore.Qt.WindowModal)
            bar.resize((self.ui.beforeView.width()), (self.ui.beforeView.width() / 20))
            bar.move(self.ui.beforeView.width(), self.ui.beforeView.width())
            bar.setMaximum(4)
            bar.show()
            QtGui.QApplication.processEvents()
            # create params to pass into align_images function
            self.alignedImages = []
            self.alignedData = []
            # x and y are intuitively switched in align_images()
            x = int(float(self.fullHeight) / (self.ui.afterView.height()) * self.selectedRect[1])
            y = int(float(self.fullWidth) / (self.ui.afterView.width()) * self.selectedRect[0])
            width = int(float(self.fullWidth) / (self.ui.afterView.width()) * self.side)
            colorlayer = self.ui.channelSelectMenu.currentIndex() - 1
            # get wiggle parameter
            try:
                wiggle = int(self.ui.alignField.text())
            except:
                wiggle = 5  # defualt wiggle, to be changed in **************** PREFERENCES
                self.ui.alignField.setText(QtCore.QString(str(5)))
            minWiggle, maxWiggle = 2, (self.selectedRect[2] - self.selectedRect[0]) / 2
            if wiggle < minWiggle:
                print 'wiggle was upgraded to min value of ', minWiggle  # **** Express this w/QDialog?
                wiggle = minWiggle
                self.ui.alignField.setText(QtCore.QString(str(minWiggle)))
            elif wiggle > maxWiggle:
                print 'wiggle was downgraded to max value of ', maxWiggle
                wiggle = maxWiggle
                self.ui.alignField.setText(QtCore.QString(str(maxWiggle)))
            if wiggle < 5:
                print 'wiggle is too small. aborting...'
                return
            # params for align_images
            print x, y, width, colorlayer, self.indexLayer, wiggle
            # make un-thresholded aligned data
            normalAligned = align_images(self.unalignedData[0], wiggle, True,
                                x, y, width, colorlayer, self.indexLayer)
            self.alignedData.append(normalAligned)
            self.ui.layerSlider.setMaximum((len(normalAligned) - 1))
            # update progress bar
            bar.setWindowTitle(QtCore.QString('Caching Aligned Images'))
            bar.setValue(1)
            QtGui.QApplication.processEvents()
            # convert un-thresholded aligned data to images
            normalAImages = []
            for datum in normalAligned:
                normalAImages.append(self.array16ToQImage(datum))
            self.alignedImages.append(normalAImages)
            # update progress bar
            bar.setWindowTitle(QtCore.QString('Aligning Images with Threshold'))
            bar.setValue(2)
            QtGui.QApplication.processEvents()
            # make thresholded aligned data
            thresholdedAligned = align_images(self.unalignedData[1], wiggle, True, x,
                y, width, colorlayer, self.indexLayer)
            self.alignedData.append(thresholdedAligned)
            # update progress bar
            bar.setWindowTitle(QtCore.QString('Caching Aligned Images with Threshold'))
            bar.setValue(3)
            QtGui.QApplication.processEvents()
            # convert thresholded aligned data to images
            threshAImages = []
            for datum in thresholdedAligned:
                threshAImages.append(self.array16ToQImage(datum))
            self.alignedImages.append(threshAImages)
            # update afterView, labels, and buttons
            self.drawAfterView()
            self.ui.afterLabel.setText(QtCore.QString('After Normalizing and Aligning'))
            self.ui.alignButton.setText(QtCore.QString('Unalign'))
            bar.close()
        else:  # already aligned, so redraw unaligned images
            self.alignedImages = []  # removed aligned data
            # update view, buttons, sliders, dropdowns, labels
            self.ui.layerSlider.setMaximum((len(self.unalignedData[0]) - 1))
            self.ui.alignButton.setText(QtCore.QString('Align'))
            self.alignMode = True
            print 'was unaligned'
            self.ui.alignField.setVisible(True)
            self.ui.channelSelectMenu.setVisible(True)
            self.ui.channelSelectMenu.setCurrentIndex(0)
            self.drawAfterView()

    def array16ToQImage(self, datum):
        '''
        :param datum: 16uint numpy array of image
        :return: img: 8bit qimage
        '''
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
        :return: none: changes drawn rectangle to nil and redraws afterView
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
        try:
            img = self.unalignedImages[0][self.indexLayer].scaled(width, height)
        except:
            print self.indexLayer, width, height, len(self.unalignedImages)
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
        width, height, thresh = self.ui.afterView.width(), self.ui.afterView.height(), 0
        if self.ui.thresholdMode.isChecked():
            thresh += 1
        if self.alignMode:
            thresh += 1
            channel = self.ui.channelSelectMenu.currentIndex()
            img = self.unalignedImages[thresh][self.indexLayer][channel].scaled(width, height)
        else:
            img = self.alignedImages[thresh][self.indexLayer].scaled(width, height)
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
        filename = str(dialog.getOpenFileName(filter=QtCore.QString('CZI File (*.czi)')))
        if not filename:  # user pressed cancel
            return
        self.filename = filename
        # initial layer displayed is 0th index
        self.indexLayer = 0  # initial layer displayed is 0th index
        self.unalignedData = []
        self.alignedImages = []
        self.unalignedImages = []
        self.alignedData = []
        # convert czi file to numpy arrays
        originalData = read_czi_file(self.filename)
        # initiate a progress bar
        bar = QtGui.QProgressBar()
        bar.setWindowTitle(QtCore.QString('Importing File...'))
        bar.setWindowModality(QtCore.Qt.WindowModal)
        bar.resize((self.ui.beforeView.width()), (self.ui.beforeView.width() / 20))
        bar.move(self.ui.beforeView.width(), self.ui.beforeView.width())
        currentProgress = 0
        bar.setMaximum((len(originalData) * 12))
        bar.show()
        QtGui.QApplication.processEvents()
        # save dimensions of imported images as attributes for later
        self.fullHeight, self.fullWidth = originalData[0].shape[0], originalData[0].shape[1]
        # convert numpy arrays to QImages and save as attribute
        originalLayers = []
        bar.setWindowTitle(QtCore.QString('Importing File...Caching Original Data'))
        QtGui.QApplication.processEvents()
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
        self.unalignedImages.append(originalLayers)  # unalignedImages[0]
        self.drawBeforeView()  # push the original image to beforeView
        bar.setWindowTitle(QtCore.QString('Importing File...Normalizing Original Data'))
        QtGui.QApplication.processEvents()
        # normalize original images without threshold and save as attribute
        normalizedData = normalize_generic(originalData)
        self.unalignedData.append(normalizedData)  # unalignedData[0]
        bar.setWindowTitle(QtCore.QString('Importing File...Caching Normalized Data (w/RGB splitting)'))
        QtGui.QApplication.processEvents()
        # convert normalized data to images
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
        self.unalignedImages.append(normalizedImages)  # unalignedImages[1]
        # normalize original images with threshold and save as attribute
        bar.setWindowTitle(QtCore.QString('Importing File...Normalizing Original Data with Threshold'))
        QtGui.QApplication.processEvents()
        thresholdVal = self.ui.thresholdMenu.currentIndex() + 1
        thresholdedData = normalize_with_standard_deviation(originalData, thresholdVal)
        # convert normalized with threshold data to images
        self.unalignedData.append(thresholdedData)  # unalignedData[1]
        bar.setWindowTitle(QtCore.QString('Importing File...Caching Normalized Data with Threshold (w/RGB splitting)'))
        QtGui.QApplication.processEvents()
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
        self.unalignedImages.append(thresholdedImages)  # unalignedImages[2]
        self.alignMode = True  # afterView is ready to draw rects with mouse
        self.drawAfterView()  # push to normalized images afterView
        # update labels, buttons, dropdowns
        self.ui.afterLabel.setText(QtCore.QString('After Normalizing'))
        self.ui.channelSelectMenu.setVisible(True)  # past normalizing, so ready to align
        self.ui.saveButton.setVisible(True)
        self.ui.saveStackButton.setVisible(True)
        self.ui.saveMIPButton.setVisible(True)
        self.ui.thresholdMode.setVisible(True)
        self.ui.thresholdMenu.setVisible(True)
        # set number of layers in image to slide through with slider widget
        self.ui.layerSlider.setMaximum((len(originalData) - 1))
        self.ui.layerSlider.setTickPosition(0)

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
        self.ui.thresholdMenu.move(x, y)
        x += self.ui.thresholdMenu.width() + 5
        self.ui.thresholdMode.move(x, y - 10)
        # adjust align and channelselect buttons
        x = width * .98 - self.ui.alignButton.width()
        self.ui.alignButton.move(x, y)
        x -= 5 + self.ui.alignField.width()
        self.ui.alignField.move(x, y + 5)
        x -= 5 + self.ui.channelSelectMenu.width()
        self.ui.channelSelectMenu.move(x, y)
        # adjust save buttons
        x = width * .98 - self.ui.saveStackButton.width()
        y = 2
        self.ui.saveStackButton.move(x, y)
        x -= self.ui.saveMIPButton.width()
        self.ui.saveMIPButton.move(x, y)
        x -= self.ui.saveButton.width()
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
