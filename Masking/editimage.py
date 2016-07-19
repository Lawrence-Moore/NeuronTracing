from PyQt4 import QtGui, QtCore
import time
import numpy as np
import cv2
import arrayfire as af

class EditWindow(QtGui.QMainWindow):
    def __init__(self, img, gpumode, bounds, parent=None):
        QtGui.QMainWindow.__init__(self)
        self.doneInitializing = False
        self.donePushing = True
        self.gpuMode = False  # gpumode is slower than cpumode right now
        # create the view
        self.view = QtGui.QGraphicsView(self)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setFrameStyle(QtGui.QFrame.NoFrame)
        self.view.show()
        self.image = img
        self.bounds = bounds
        self.w2hratio = 5  # in the sliders
        self.colors = ['R: ', 'G: ', 'B: ']
        self.sliders = []
        self.rSlider = QtGui.QGraphicsView(self)
        self.gSlider = QtGui.QGraphicsView(self)
        self.bSlider = QtGui.QGraphicsView(self)
        for slider in [self.rSlider, self.gSlider, self.bSlider]:
            slider.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            slider.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            slider.setFrameStyle(QtGui.QFrame.NoFrame)
            slider.viewport().installEventFilter(self)
            slider.viewport().setMouseTracking(True)
            self.sliders.append(slider)
        self.sliderViewPoints = [[], [], []]
        self.rLabel = QtGui.QLabel(self)
        self.gLabel = QtGui.QLabel(self)
        self.bLabel = QtGui.QLabel(self)
        self.labels = []
        for color, label in enumerate([self.rLabel, self.gLabel, self.bLabel]):
            label.setText(QtCore.QString('%s%d' % (self.colors[color],
                                                   self.bounds[color][1])))
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.show()
            self.labels.append(label)
        # rgb fields
        self.rMaxField = QtGui.QTextEdit(self)
        self.gMaxField = QtGui.QTextEdit(self)
        self.bMaxField = QtGui.QTextEdit(self)
        self.rMinField = QtGui.QTextEdit(self)
        self.gMinField = QtGui.QTextEdit(self)
        self.bMinField = QtGui.QTextEdit(self)
        self.fields = []
        for color, field in enumerate([self.rMaxField, self.gMaxField, self.bMaxField,
                      self.rMinField, self.gMinField, self.bMinField]):
            field.setText(QtCore.QString(str(self.bounds[color/2][0])))
            field.setAlignment(QtCore.Qt.AlignRight)
            field.setFrameStyle(QtGui.QFrame.NoFrame)
            field.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            field.installEventFilter(self)
            #field.textChanged.connect(self.fieldUpdate)
            #field.enterEvent.connect(self.fieldUpdate)
            field.show()
            self.fields.append(field)
        for color, field in enumerate([self.rMaxField, self.gMaxField, self.bMaxField]):
            field.setText(QtCore.QString('%d' % self.bounds[color][2]))
        self.rCheck = QtGui.QCheckBox(self)
        self.gCheck = QtGui.QCheckBox(self)
        self.bCheck = QtGui.QCheckBox(self)
        self.checks = []
        for check in [self.rCheck, self.gCheck, self.bCheck]:
            check.setChecked(True)
            check.show()
            self.checks.append(check)
            check.stateChanged.connect(self.rangesRelay)
        self.image2View(self.image)
        # create save and reset buttons
        self.saveImageButton = QtGui.QPushButton(self)
        self.saveImageButton.setText('Save to Main')
        self.resetButton = QtGui.QPushButton(self)
        self.resetButton.setText('Reset')
        self.resetButton.released.connect(self.resetBounds)
        self.activeSlider = 0  # min/mid/max
        self.mouseHold = False
        self.doneInitializing = True

    def rangesRelay(self):
        if self.doneInitializing:
            self.ranges2View()

    def eventFilter(self, source, event):
        # respond to mouse events in colorSpace after importing file to mipViews
        if source == self.rSlider.viewport() or self.gSlider.viewport() or self.bSlider.viewport():
            if source == self.rSlider.viewport():
                color = 0
            elif source == self.gSlider.viewport():
                color = 1
            else:
                color = 2
            if event.type() == QtCore.QEvent.MouseButtonPress:
                pos = event.pos()
                self.getSlider(color, pos.x(), pos.y())
            elif event.type() == QtCore.QEvent.MouseMove and self.mouseHold:
                pos = event.pos()
                self.moveSlider(color, pos.y())
            elif event.type() == QtCore.QEvent.MouseButtonRelease and self.mouseHold:
                self.mouseHold = False
                pos = event.pos()
                self.moveSlider(color, pos.y())
        if event.type() == QtCore.QEvent.MetaCall and source in self.fields:
            self.fieldUpdate()
        return False

    def resetBounds(self):
        self.bounds = [[0, 127, 255], [0, 127, 255], [0, 127, 255]]
        # set the text fields, and sliders to the appropriate value
        for color, (label, maxField, minField) in enumerate([(self.rLabel,
                self.rMaxField, self.rMinField), (self.gLabel,
                self.gMaxField, self.gMinField), (self.bLabel,
                self.bMaxField, self.bMinField)]):
            minField.blockSignals(True) ######################################## remove this if signals from min/max replaced by enter
            minField.setText(QtCore.QString(str(self.bounds[color][0])))
            minField.blockSignals(False)
            label.blockSignals(True)
            label.setText(QtCore.QString('%s%d' % (self.colors[color],
                                                   self.bounds[color][1])))
            label.blockSignals(False)
            maxField.blockSignals(True)
            maxField.setText(str(self.bounds[color][2]))
            maxField.blockSignals(False)
        for c in xrange(0, 3):
            self.createSliderView(c)
        self.ranges2View()

    def resizeEvent(self, event):
        width, height = event.size().width(), event.size().height()
        self.remakeLayout(width, height)

    def getSlider(self, color, px, py):
        for p, pos in enumerate(self.sliderViewPoints[color]):
            for [x, y] in pos:
                if px == x and py == y:
                    self.activeSlider = p
                    self.mouseHold = True
                    self.autoRefresh()
                    return

    def autoRefresh(self):
        while self.mouseHold:
            time.sleep(0.03)
            self.ranges2View()

    def createSliderView(self, c):
        slider = self.sliders[c]
        width, height = self.rSlider.width(), self.rSlider.height()
        img = QtGui.QImage(width, height, QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.qRgba(0, 0, 0, 0))
        fillColor = QtGui.qRgba(255, 255, 255, 255)
        yshift = width / self.w2hratio  # so that 255 and 0 can be easily targeted
        self.sliderViewPoints[c] = []
        for pos in self.bounds[c]:
            mid = int(float(height - 2 * yshift) * (1. - float(pos) / 255.)) + yshift
            lst = []
            for x in xrange(0, width):
                dy = (width - x) / self.w2hratio
                for y in xrange((mid - dy), (mid + dy)):
                    if not (0 < x < width and 0 < y < height):
                        continue
                    img.setPixel(x, y, fillColor)
                    lst.append([x, y])
            self.sliderViewPoints[c].append(lst)
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, width, height)
        gradient = self.makeGradient(c, yshift)
        pic = QtGui.QPixmap.fromImage(gradient)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        pic = QtGui.QPixmap.fromImage(img)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))
        slider.setScene(scene)
        slider.setRenderHint(QtGui.QPainter.Antialiasing)
        slider.show()
        del img

    def makeGradient(self, color, yshift):
        [i, m, f] = self.bounds[color]
        width, height = self.rSlider.width(), self.rSlider.height()
        grad = np.zeros((height, width, 3), dtype=np.uint8)
        ocolor, oocolor = (color + 1) % 3, (color + 2) % 3
        newheight = height - 2 * yshift
        i = height - int(i * (newheight / 255.))
        m = height - int(m * (newheight / 255.))
        f = height - int(f * (newheight / 255.))
        initial = i - yshift
        middle = m - yshift
        final = f - yshift
        fracColor = 255
        for y in xrange(0, final):
            grad[y, :, color] = np.full((width), 255, dtype=np.uint8)
            grad[y, :, ocolor] = np.full((width), 0, dtype=np.uint8)
            grad[y, :, oocolor] = np.full((width), 0, dtype=np.uint8)
        for y in xrange(final, middle):
            grad[y, :, color] = np.full((width), int(fracColor), dtype=np.uint8)
            grad[y, :, ocolor] = np.full((width), 0, dtype=np.uint8)
            grad[y, :, oocolor] = np.full((width), 0, dtype=np.uint8)
            fracColor += 128. / (f - m)
        for y in xrange(middle, initial):
            grad[y, :, color] = np.full((width), int(fracColor), dtype=np.uint8)
            grad[y, :, ocolor] = np.full((width), 0, dtype=np.uint8)
            grad[y, :, oocolor] = np.full((width), 0, dtype=np.uint8)
            fracColor += 128. / (m - i)
        for y in xrange(initial, height):
            grad[y] = np.full((width, 3), 0, dtype=np.uint8)
        img = QtGui.QImage(grad, grad.shape[1], grad.shape[0],
                            grad.shape[1] * 3, QtGui.QImage.Format_RGB888)
        return img

    def fieldUpdate(self):
        if not self.doneInitializing:
            return
        for color, (maxField, minField) in enumerate([(self.rMaxField, self.rMinField),
                (self.gMaxField, self.gMinField), (self.bMaxField, self.bMinField)]):
            if str(minField.toPlainText()) == '' or str(maxField.toPlainText()) == '':
                continue
            try:
                self.bounds[color][0] = int(minField.toPlainText())
            except:
                self.fieldError(maxField.toPlainText())
            try:
                self.bounds[color][2] = int(maxField.toPlainText())
            except:
                self.fieldError(maxField.toPlainText())
            if self.bounds[color][0] > 253:
                self.bounds[color][0] = 252
            if self.bounds[color][2] < 2:
                self.bounds[color][2] = 3
            if self.bounds[color][2] > 255:
                self.bounds[color][2] = 255
            if self.bounds[color][0] < 0:
                self.bounds[color][0] = 0
            self.bounds[color][1] = (self.bounds[color][2] + self.bounds[color][0]) / 2
            if self.bounds[color][0] >= self.bounds[color][2]:
                self.bounds[color][0] = self.bounds[color][1] - 1
                self.bounds[color][2] = self.bounds[color][1] + 1
            maxField.setText(QtCore.QString(str(self.bounds[color][2])))
            maxField.moveCursor(QtGui.QTextCursor.End)
            minField.setText(QtCore.QString(str(self.bounds[color][0])))
            minField.moveCursor(QtGui.QTextCursor.End)
            self.createSliderView(color)
            self.labels[color].setText(QtCore.QString(self.colors[color] + str(self.bounds[color][1])))
        self.ranges2View()

    def moveSlider(self, color, py):
        width, height = self.rSlider.width(), self.rSlider.height()
        yshift = width / self.w2hratio
        y = int(((height - py - yshift) / float(height - 2 * yshift)) * 255)
        if self.activeSlider == 0:
            if y > 253 or y < 0 or y >= self.bounds[color][1]:
                return
            self.fields[3+color].setText(QtCore.QString(str(y)))
        elif self.activeSlider == 2:
            if y < 2 or y > 255 or y <= self.bounds[color][1]:
                return
            self.fields[color].setText(QtCore.QString(str(y)))
        elif self.activeSlider == 1:
            if y >= self.bounds[color][2] or y <= self.bounds[color][0]:
                return
            self.labels[color].setText(QtCore.QString(self.colors[color] + str(y)))
        self.bounds[color][self.activeSlider] = y
        self.createSliderView(color)
        self.ranges2View()

    def ranges2View(self, unscaled=False):
        if not self.donePushing:
            return
        a = time.time()
        if unscaled:
            adjusted = self.image.copy()
        else:
            self.donePushing = False
            adjusted = cv2.resize(self.image, (self.view.width(), self.view.height()))
        r, g, b = adjusted[:, :, 0], adjusted[:, :, 1], adjusted[:, :, 2]
        for c, color in enumerate([r, g, b]):
            height, width = color.shape[0], color.shape[1]
            if not self.checks[c].isChecked():
                adjusted[:, :, c] = np.zeros((height, width))
                continue
            [i, m, f] = self.bounds[c]
            i, m, f = i * 256, m * 256, f * 256
            maximum = 65535  # in 16-bit unsigned
            color[color < i] = i
            color[color > f] = f
            predictedMid = (i + f) / 2
            if self.gpuMode:
                color = color.reshape((width * height))
            overmid = color.copy()
            overmidmask = (overmid >= m)
            if self.gpuMode:
                overmid = af.interop.np_to_af_array(overmid)
                for ii in af.ParallelRange(width*height):
                    overmid[ii] -= m
                    overmid[ii] *= (float(maximum - predictedMid) / (f - m))
                    overmid[ii] += predictedMid
                overmid = np.array(overmid)
                overmid = overmid.reshape(height, width)
            else:
                overmid = ((overmid-m) * (float(maximum - predictedMid) / (f - m))) + predictedMid
            overmid *= overmidmask
            undermid = color
            undermidmask = (undermid < m)
            if self.gpuMode:
                undermid = af.interop.np_to_af_array(undermid)
                for ii in af.ParallelRange(width*height):
                    undermid[ii] -= i
                    undermid[ii] *= (float(predictedMid - i) / (m - i))
                undermid = np.array(undermid)
                undermid = undermid.reshape(height, width)
            else:
                undermid -= i
                undermid *= (float(predictedMid - i) / (m - i))
            undermid *= undermidmask
            color = undermid + overmid
            adjusted[:, :, c] = color
        adjusted /= 256
        adjusted = adjusted.astype(np.uint8)
        if unscaled:
            self.returnImage = adjusted
        else:
            self.image2View(adjusted, scaled=True)
        b = time.time()
        print 'color rescaling ms', 1000*(b-a)

    def image2View(self, img, scaled=False):
        # convert from numpy array to QImage
        img = QtGui.QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3,
                           QtGui.QImage.Format_RGB888)
        if not scaled:
            img = img.scaled(self.view.width(), self.view.height())
        # create the graphics scene
        scene = QtGui.QGraphicsScene()
        scene.setSceneRect(0, 0, self.view.width(), self.view.height())
        pic = QtGui.QPixmap.fromImage(img)
        scene.addItem(QtGui.QGraphicsPixmapItem(pic))  # add image to scene
        # push scene to dynamicView's graphics window
        self.view.setScene(scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.show()
        del img
        QtGui.QApplication.processEvents()
        self.donePushing = True

    def fieldError(self, text):
        error = QtGui.QMessageBox()
        error.setText(QtCore.QString('Error! Only integers can be entered into'
                                     ' the text fields, not %s.' % text))
        error.exec_()

    def remakeLayout(self, width, height):
        # the view
        viewMargin = height / 20
        self.view.move(viewMargin, viewMargin)
        viewSide = height - 2*viewMargin
        self.view.resize(viewSide, viewSide)
        # rgb sliders, labels, and min/max text fields
        interSliderMargin = 120
        fieldW, fieldH = 40, 20
        sliderWidth, sliderHeight = viewSide / 10, viewSide
        labelBufferX, labelBufferY = fieldW / 2, 22
        x = viewMargin + viewSide + interSliderMargin / 2
        if sliderHeight > 400:
            sliderHeight = 400
            sliderWidth = 40
        fieldBufferX, fieldY = 50, sliderHeight + viewMargin - fieldH
        for slider, label, maxField, minField, check in [(self.rSlider, self.rLabel,
                self.rMaxField, self.rMinField, self.rCheck), (self.gSlider, self.gLabel,
                self.gMaxField, self.gMinField, self.gCheck), (self.bSlider, self.bLabel,
                self.bMaxField, self.bMinField, self.bCheck)]:
            slider.resize(sliderWidth, sliderHeight)
            slider.move(x, viewMargin)
            maxField.move(x - fieldBufferX, viewMargin)
            minField.move(x - fieldBufferX, fieldY)
            maxField.resize(fieldW, fieldH)
            minField.resize(fieldW, fieldH)
            label.move(x-fieldW+sliderWidth/2, viewMargin-labelBufferY)
            label.resize(fieldW*2, fieldH)
            check.move(x, viewMargin+sliderHeight)
            x += interSliderMargin
            slider.raise_()
        w, h = .1*width, .04*height
        self.saveImageButton.resize(w, h)
        self.resetButton.resize(w, h)
        x, bufferY = (.98 * width - self.saveImageButton.width()), 0.08 * height
        self.saveImageButton.move(x, height-bufferY)
        self.resetButton.move(x, bufferY - self.resetButton.height())
        for c in xrange(0, 3):
            self.createSliderView(c)
        self.ranges2View()

    def returnEditedImage(self):
        self.ranges2View(True)
        return self.returnImage, self.bounds

