from PyQt4 import QtGui, QtCore
import sys
from mainwindow import Ui_MainWindow
from colorspace import colorSpaces
from mip import mips
import time
import arrayfire as af

#print af.info()
sys.setrecursionlimit(50000)  # for packaging into OSX

# Docstring Format:
# Param ArgName: (ArgType:) Description
# Param ArgName2: (ArgType2:) Description
# Return: (ReturnType:) Description

# Comments precede what they describe unless on same line or continuing.
# Variable description often as "type: description"

class Run(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        try:
            af.info()
            self.ui.gpuLabel.setText(QtCore.QString('GPU: ON'))
            self.gpuMode = True
        except:
            self.gpuMode = False
        # create color space, attach signal: slider, add event to filter
        self.colorSpace = colorSpaces(self.ui.colorSpace,
            self.ui.intensityLabel, self.ui.intensitySlider, self.ui.volumeSelect,
            self.ui.modeMenu, self.ui.areaSelectionView)
        self.ui.intensitySlider.valueChanged.connect(self.colorSpace.updateColorSpaceView)
        self.ui.colorSpace.viewport().installEventFilter(self)
        self.ui.colorSpace.viewport().setMouseTracking(True)
        self.ui.modeMenu.activated[str].connect(self.colorSpace.modeChange)
        self.ui.deleteVolume.clicked.connect(self.colorSpace.deleteVolume)
        self.ui.addAreaButton.clicked.connect(self.colorSpace.addArea)
        self.ui.delAreaButton.clicked.connect(self.colorSpace.deleteArea)
        self.ui.volumeSelect.activated[str].connect(self.colorSpace.volumeChange) # handles multiple color-space volumes
        self.ui.areaSelectionView.viewport().installEventFilter(self)
        # add signals for mipDynamic view
        self.ui.mipDynamic.viewport().installEventFilter(self)
        self.ui.mipDynamic.viewport().setMouseTracking(True)
        # create mip views
        self.mipViews = mips(self.ui.mipFull, self.ui.mipDynamic, self.ui.colorSpace, self.ui.editImageButton, self.gpuMode)
        self.colorSpace.addDynamicView(self.mipViews.updateDynamic)
        self.ui.plusZoom.pressed.connect(self.mipViews.zoomPlus)
        self.ui.minusZoom.pressed.connect(self.mipViews.zoomMinus)
        self.ui.mipFull.viewport().installEventFilter(self)
        self.ui.mipFull.viewport().setMouseTracking(True)
        # controls
        self.ui.importImageButton.released.connect(self.mipViews.importImage)
        self.remakeLayout(self.width(), self.height())
        # save images
        self.ui.saveDynamicButton.released.connect(self.mipViews.saveImage)
        self.ui.applyStackButton.released.connect(self.savingStack)
        self.ui.editImageButton.released.connect(self.mipViews.editImage)

    def resizeEvent(self, event):
        width, height = event.size().width(), event.size().height()
        self.remakeLayout(width, height)

    def savingStack(self):
        self.colorSpace.saveStack(self.mipViews.originalImage, self.mipViews.imageBeforeEditing)

    def remakeLayout(self, width, height):
        '''
        :param width:
        :param height:
        :return: This application does not support resizing of areas after they
        are drawn.
        '''
        # importImageButton
        x = 10
        y = 20
        self.ui.importImageButton.move(x, y)
        x += self.ui.importImageButton.width() + 5
        # mipViews
        self.ui.saveDynamicButton.move(x, y)
        x += self.ui.saveDynamicButton.width() + 5
        self.ui.applyStackButton.move(x, y)
        x = width / 2
        y = 15
        side = (height / 2) - 30
        self.ui.mipFull.move(x, y)
        self.ui.mipFull.resize(side, side)
        self.ui.mipDynamic.move(x, (height / 2))
        self.ui.mipDynamic.resize(side, side)
        x += side + 10
        bside, bseparation = side * .07, side * .1
        self.ui.editImageButton.move(x, y)
        self.ui.editImageButton.resize(bside, bside)
        y += bseparation * 2
        self.ui.plusZoom.move(x, y)
        self.ui.plusZoom.resize(bside, bside)
        y += bseparation
        self.ui.minusZoom.move(x, y)
        self.ui.minusZoom.resize(bside, bside)
        # mode and volume controls
        y = int(height * .12)
        self.ui.modeMenu.move(int(width * .03), y)
        x = int(width * .25)
        self.ui.volumeLabel.move(x, y)
        x += self.ui.volumeLabel.width() + 5
        self.ui.volumeSelect.move(x, y)
        x += self.ui.volumeSelect.width() + 5
        self.ui.deleteVolume.move(x, y)
        y = height / 4 - 20
        side = height * .4
        self.ui.colorSpaceLabel.move(20, y)
        self.ui.colorSpaceLabel.resize(side, 20)
        y = height / 4
        '''
        scale = float(side) / self.ui.colorSpace.width()
        for i in xrange(0, len(self.colorSpace.volumes)): # [areas, viewpoints, drawingareas, currentarea]
            if type(self.colorSpace.volumes[i]) is bool:
                continue
            for ii in xrange(0, len(self.colorSpace.volumes[i][0])): # [area, area]
                if type(self.colorSpace.volumes[i][0]) is bool:
                    continue
                for iii in xrange(0, len(self.colorSpace.volumes[i][0][ii][0])):  # [x, y]
                    self.colorSpace.volumes[i][0][ii][0][iii] *= scale
                for iii in xrange(0, len(self.colorSpace.volumes[i][0][ii][1])):  # [x, y]
                    self.colorSpace.volumes[i][0][ii][1][iii] *= scale
            for ii in xrange(0, len(self.colorSpace.volumes[i][2])):  # [x, y]
                if type(self.colorSpace.volumes[i][2][ii]) is bool:
                    continue
                for iii in xrange(0, len(self.colorSpace.volumes[i][2][ii][1])):
                    self.colorSpace.volumes[i][2][ii][1][iii][0] *= scale
                    self.colorSpace.volumes[i][2][ii][1][iii][1] *= scale
            for ii in xrange(0, len(self.colorSpace.volumes[i][3])):
                try:
                    self.colorSpace.volumes[i][3][ii][0] *= scale
                    self.colorSpace.volumes[i][3][ii][1] *= scale
                except:
                    print self.colorSpace.volumes[i][3][ii]
        for i in xrange(0, len(self.colorSpace.currentArea)):
            self.colorSpace.currentArea[i][0] *= scale
            self.colorSpace.currentArea[i][1] *= scale
        '''
        self.ui.colorSpace.move(20, y)
        self.ui.colorSpace.resize(side, side)
        x = side + 30
        w = int(.02*width)
        self.ui.areaSelectionView.move(x, y - w/4)
        self.ui.areaSelectionView.resize(w, side + w/2)
        x += int(.02*width)-10
        self.ui.intensitySlider.move(x, y)
        self.ui.intensitySlider.resize(22, side)
        y -= w/4 + 20
        self.ui.intensityLabel.move(x - 30 - (self.ui.intensitySlider.width() / 2), y)
        x = int(width * .22)
        y -= 35
        self.ui.addAreaButton.move(x, y)
        x += self.ui.addAreaButton.width() + 5
        self.ui.delAreaButton.move(x, y)
        y = height / 4 + side + 20
        self.ui.debugLabel.move(20, y)
        self.ui.debugLabel.resize((width / 2 - 100), (height - y - 50))
        self.ui.gpuLabel.move((width - self.ui.gpuLabel.width()), (height - self.ui.gpuLabel.height()))
        # remake colorspace view if nothing is drawn (won't resize drawings)
        if self.colorSpace.areas == [False] and self.colorSpace.volumes == [False]:
            self.colorSpace.createColorSpaceView()
        # resize the mip views
        if self.mipViews.filename:
            self.mipViews.updateMipView()
        self.ui.intensityLabel.setTextFormat(QtCore.Qt.PlainText)
        if side > 400:  # change size of node drawn in colorSpace
            self.colorSpace.nodeSize = [int(-1 * side / 200), int((side / 200) + 1)]

    def keyPressEvent(self, event):
        key = event.key()
        # print event.text()
        if key == QtCore.Qt.Key_Escape:
            self.close()
            quit()
        if not self.mipViews.filename:
            return
        elif key == QtCore.Qt.Key_Minus:
            self.mipViews.zoomMinus()
        elif key == QtCore.Qt.Key_Equal:
            self.mipViews.zoomPlus()
        elif key == QtCore.Qt.Key_Left or key == QtCore.Qt.Key_A:
            self.mipViews.shift('l')
        elif key == QtCore.Qt.Key_Right or key == QtCore.Qt.Key_D:
            self.mipViews.shift('r')
        elif key == QtCore.Qt.Key_Up or key == QtCore.Qt.Key_W:
            self.mipViews.shift('u')
        elif key == QtCore.Qt.Key_Down or key == QtCore.Qt.Key_S:
            self.mipViews.shift('d')
        elif key == QtCore.Qt.Key_C and self.colorSpace.areaMode == 'manualUpdate':
            radius = self.colorSpace.side / 2
            self.colorSpace.updateManualBoundary(radius, radius, False)


    def eventFilter(self, source, event):
        # respond to mouse events in colorSpace after importing file to mipViews
        if source == self.ui.colorSpace.viewport():
            try:  # if nothing has been imported yet
                if not self.mipViews.filename:
                    return False
            except:  # if mipViews hasn't been created yet
                return False
            if self.colorSpace.areaMode == 'auto':
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    self.colorSpace.mouseHold = True
                    pos = event.pos()
                    self.colorSpace.createBoundary(pos.x(), pos.y())
                elif event.type() == QtCore.QEvent.MouseMove and self.colorSpace.mouseHold:
                    pos = event.pos()
                    self.colorSpace.createBoundary(pos.x(), pos.y())
                elif event.type() == QtCore.QEvent.MouseButtonRelease:
                    self.colorSpace.mouseHold = False
                    pos = event.pos()
                    self.colorSpace.createBoundary(pos.x(), pos.y())
                    self.colorSpace.finalizeBoundary()
                    self.colorSpace.createAreaView()
            elif self.colorSpace.areaMode == 'manualCreate':
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    pos = event.pos()
                    self.colorSpace.createManualBoundary(pos.x(), pos.y())
                    self.colorSpace.createAreaView()
            elif self.colorSpace.areaMode == 'manualUpdate':
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    self.colorSpace.mouseHold = True
                    pos = event.pos()
                    self.colorSpace.updateManualBoundary(pos.x(), pos.y(), True)
                elif event.type() == QtCore.QEvent.MouseMove and self.colorSpace.mouseHold and self.colorSpace.doneDrawing:
                    self.colorSpace.doneDrawing = False
                    pos = event.pos()
                    self.colorSpace.updateManualBoundary(pos.x(), pos.y(), False)
                    QtGui.QApplication.processEvents()
                    self.colorSpace.doneDrawing = True
                elif event.type() == QtCore.QEvent.MouseButtonRelease:
                    self.colorSpace.mouseHold = False
            elif self.colorSpace.areaMode == 'circular':
                if event.type() == QtCore.QEvent.MouseButtonPress:
                    self.colorSpace.mouseHold = True
                    pos = event.pos()
                    self.colorSpace.createBoundary(pos.x(), pos.y())
                elif event.type() == QtCore.QEvent.MouseMove and self.colorSpace.mouseHold:
                    pos = event.pos()
                    self.colorSpace.circularRadius(pos.x(), pos.y())
                elif event.type() == QtCore.QEvent.MouseButtonRelease:
                    self.colorSpace.mouseHold = False
                    pos = event.pos()
                    self.colorSpace.circularRadius(pos.x(), pos.y())
                    self.colorSpace.circularCreate()
                    self.colorSpace.createAreaView()
        elif source == self.ui.areaSelectionView.viewport() and event.type()\
                == QtCore.QEvent.MouseButtonPress:
            pos = event.pos()
            self.colorSpace.getPreviousArea(pos.x(), pos.y())
        elif source == self.ui.mipFull.viewport() and event.type() == QtCore.QEvent.MouseButtonPress:
            pos = event.pos()
            xyv = self.mipViews.mip2ColorSpace(pos.x(), pos.y())
            self.colorSpace.circularFromMip(xyv)
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.debugLog()
        return False

    def debugLog(self):
        st = '# of areas: ' + str(len(self.colorSpace.areas))
        st += ', # of vertices in current area: ' + str(len(self.colorSpace.currentArea))
        st += '\n# of volumes: ' + str(len(self.colorSpace.volumes))
        st += '\narea mode: ' + self.colorSpace.areaMode
        st += '\nindexes: area, ' + str(self.colorSpace.indexArea)
        st += '. volume, ' + str(self.colorSpace.indexVolume)
        self.ui.debugLabel.setText(QtCore.QString(st))

if __name__ == '__main__':  # i added this
    app = QtGui.QApplication(sys.argv)
    ex = Run()  # this is customized
    ex.show()
    sys.exit(app.exec_())




# change log:
# 6/20: installed needed environments, modules, software
# 6/21: setup simple interface with graphics and slider widgets
# 6/21: made slider widget auto-update
# 6/22: made graphics windows respond to mouse events
# 6/22: allowed user to draw loop on color space, trace points, and compute area
# 6/22: drawing area on color space fails 1/4 times. will debug later
# 6/23: created volume from selected areas in colorspace (simple, 1-area volume)
# 6/23: created mapping from mip's image to colorspace's volume in mip
# 6/23: the dynamic mip image is partially working; still has bugs
# 6/24: fixed bug in drawing areas and issue in colorspace areas conversion
# 6/24: fully works with 2-D image and a colorspace generated from 1-area
# 6/26: added polygon/manual adding of areas to colorspace view
# 6/27: created interface to add multiple volumes in colorspace
# 6/27: made drawing tool adapt to given intensity (black or white loop)
# 6/27: allowed changing intensity value without adding area (dynamically)
# 6/28: added circular/manual adding of areas to colorspace view
# 6/28: created 2-D complex version of color-space with single volume
# 6/29: made zoom in/out work. compared gpu processing modules for python
# 6/30: learned basic operations in gpu module: arrayfire
# 7/1: implemented moving left/right/up/down on zoomed views of mip
# 7/1: implemented multiple volumes, rewrote many colorSpace algorithms with
    # a different module, since PIL to QImage was not working on windows
# 7/5: made gui dynamically change size with rescaling, made import image button
# 7/5: switched to using numpy array for editing individual pixels
# 7/6: created gui (CorrectionWin) to normalize and align images
# 7/7: commented CorrectWin and mip modules
# 7/7: created progress bars inside both CorrectWin GUI and main GUI for import
# 7/8: commented mip module, sped up color space conversion with matrix
    # multiplication as well as remaking the view in main GUI
# 7/11: made saving to-stack operations for both GUIs
# 7/12: removed dynamic updating of mipDynamic (too slow),
# 7/12: added more progressbars, saving masked Images to stack and dilating
# 7/13: created interface to change r, g, b ranges on image
    # question: doesn't this mess w/the colors hues, etc. that are going to be masked in the other images?
# 7/14: implemented frame dropping in GUIs, used arrayfire to speed up image editing
# 7/14: made circular mode autobound, framed mip views

# time-performance log:
# 6/24: 2D Simple with 200^2 pixels: (MappedMip: 470ms), (Updating: 35ms)
    # Estimated Updating in 3D-Simple with 300^3px is 24s
# 6/28: 2D Complex with 200^2 pixels: (MappedMip: 175ms), (Updating: 20ms)
    # Replaced matplotlib functions in MappedMip with own
    # Updating method switch to manual polygon. (Total slider refresh: 35ms)

# to-do:
# convert colorspace window from (side, 256, side) to (256, 256, 256) and revert back to (256, 256, 2) where 2 = range of xyv, xyv
# get checking validity map down to under 100ms (creating the map is 40ms at most)
# use numexpr with original thresholding idea (test2)?
# make "isolation" into 3D image. then mip it.
# add multithreading for saving stack
# combine 2 guis into one file
# add preference pane
# show map of where you are in the image when zoomed
########## make backgrounds for sliders for intuition