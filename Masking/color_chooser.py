from PyQt4 import QtGui, QtCore
import sys
import saving_and_color
import numpy as np
import time
import copy


class ColorChooser(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self)
        self.doneInitializing = False
        self.width, self.height = 800, 400
        self.setGeometry(QtCore.QRect(200, 200, self.width, self.height))
        # rgb = saving_and_color.xyvLst2rgb(xyv, 150., 'hsv')
        self.rgbList = [[200, 100, 100], [100, 200, 200], [200, 200, 100],
                        [200, 100, 100], [100, 200, 200], [200, 200, 100],
                        [200, 100, 100], [100, 200, 200], [200, 200, 100]]
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
        self.reset()
        self.doneInitializing = True
        # create merge buttons

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
        self.viewSize = self.height / 4
        self.reset()

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
        self.color2View(merged, self.colorViews[first])

    def delete(self):
        # this does not reposition color views!
        while True in self.selectedViews:
            i = self.selectedViews.index(True)
            self.colorViews[i].close()
            del self.colorViewPorts[i]
            del self.selectedViews[i]
            del self.colorViews[i]
            del self.modifiedRgbList[i]

    def reset(self):
        if self.doneInitializing:
            for view in self.colorViews:
                view.close()
        self.colorViews = []
        self.colorViewPorts = []
        self.selectedViews = [False] * len(self.rgbList)
        initX = 0.5 * self.viewSize
        x, y = initX, 0.5 * self.viewSize
        for i, [r, g, b] in enumerate(self.rgbList):
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


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = ColorChooser()
    ex.show()
    sys.exit(app.exec_())

