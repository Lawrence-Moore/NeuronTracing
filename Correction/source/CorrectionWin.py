from PyQt4 import QtGui, QtCore
import sys
from correction import Ui_CorrectionWindow
from image_normalization import *
from PIL import Image, ImageQt
import time
# import normalizing functions

class Correction(QtGui.QMainWindow):
    def __init__(self):
        self.drawRectMode = False  # initialize as something else!
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_CorrectionWindow()
        self.ui.setupUi(self)
        self.ui.afterView.viewport().installEventFilter(self)
        self.ui.afterView.viewport().setMouseTracking(True)
        self.selectedRect = [0, 0, 0, 0]
        self.mouseHold = False
        self.ui.importButton.pressed.connect(self.importImage)
        self.originalLayers = []
        self.normalizedLayers = []
        self.normalizedThresholdLayers = []
        self.indexLayer = 0  # initial layer displayed is 0th index
        self.ui.layerSlider.setTickPosition(self.indexLayer)
        # self.afterPixMax
        # create pen for after-view
        self.filename = ''
        self.pen = QtGui.QPen()
        self.pen.setColor(QtGui.QColor(255, 255, 255))
        self.ui.thresholdMode.stateChanged.connect(self.thresholdChanged)
        self.ui.layerSlider.valueChanged.connect(self.layerChanged)
        self.ui.layerLabel.setText(QtCore.QString('Layer: %d' % self.indexLayer))

    def thresholdChanged(self):
        self.drawBeforeView()
        self.drawAfterView()

    def layerChanged(self):
        self.indexLayer = self.ui.layerSlider.sliderPosition()
        self.ui.layerLabel.setText(QtCore.QString('Layer: %d' % self.indexLayer))
        self.drawBeforeView()
        self.drawAfterView()

    def eraseRect(self):
        self.selectedRect = [0, 0, 0, 0]
        self.drawAfterView()

    def drawBeforeView(self):
        if not self.filename:  # nothing was imported
            return
        # create scene from original image
        width, height = self.ui.beforeView.width(), self.ui.beforeView.height()
        img = self.originalLayers[self.indexLayer].scaled(width, height)
        pixmap = QtGui.QPixmap.fromImage(img)
        pixitem = QtGui.QGraphicsPixmapItem(pixmap)
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.addItem(pixitem)
        # push scene to before-view
        self.ui.beforeView.setScene(scene)
        self.ui.beforeView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.beforeView.show()

    def drawAfterView(self):
        if not self.filename:  # nothing was imported
            return
        # create scene from normalized image
        width, height = self.ui.afterView.width(), self.ui.afterView.height()
        if self.ui.thresholdMode.isChecked():
            img = self.normalizedThresholdLayers[self.indexLayer].scaled(width, height)
        else:
            img = self.normalizedLayers[self.indexLayer].scaled(width, height)
        pixmap = QtGui.QPixmap.fromImage(img)
        pixitem = QtGui.QGraphicsPixmapItem(pixmap)
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        scene.addItem(pixitem)
        if self.selectedRect != [0, 0, 0, 0]:
            [xi, yi, xf, yf] = self.selectedRect
            rect = QtCore.QRectF(xi, yi, xf-xi, yf-yi)
            scene.addRect(rect, self.pen, QtGui.QBrush())
        # push scene to after-view
        self.ui.afterView.setScene(scene)
        self.ui.afterView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.afterView.show()

    def importImage(self):
        dialog = QtGui.QFileDialog()
        self.filename = str(dialog.getOpenFileName())
        if not self.filename:  # user pressed cancel
            return
        originalDatum = read_czi_file(self.filename)
        self.ui.layerSlider.setMaximum((len(originalDatum) - 1))
        for data in originalDatum:
            data = data / 256
            data = data.astype(np.uint8)
            img = Image.fromarray(data)
            img = ImageQt.ImageQt(img)
            self.originalLayers.append(img)
        self.drawBeforeView()
        # normalize original images
        normalizedDatum = normalize_colors(originalDatum)
        for data in normalizedDatum:
            data = data / 256
            data = data.astype(np.uint8)
            img = Image.fromarray(data)
            img = ImageQt.ImageQt(img)
            self.normalizedLayers.append(img)
        thresholdedDatum = normalize_colors(originalDatum, True)
        for data in thresholdedDatum:
            data = data / 256
            data = data.astype(np.uint8)
            img = Image.fromarray(data)
            img = ImageQt.ImageQt(img)
            self.normalizedThresholdLayers.append(img)
        # push to after-view
        self.drawRectMode = True
        self.drawAfterView()

    def eventFilter(self, source, event):
        if source == self.ui.afterView.viewport() and self.drawRectMode:
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
        key = event.key()
        if key == QtCore.Qt.Key_Delete:
            self.eraseRect()

    def resizeEvent(self, event):
        width, height = event.size().width(), event.size().height()
        self.remakeLayout(width, height)

    def remakeLayout(self, width, height):
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
        self.drawBeforeView()
        self.drawAfterView()


if __name__ == '__main__':  # i added this
    app = QtGui.QApplication(sys.argv)
    ex = Correction()  # this is customized
    ex.show()
    sys.exit(app.exec_())
