# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1005, 768)
        self.centralWidget = QtGui.QWidget(MainWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.mipFull = QtGui.QGraphicsView(self.centralWidget)
        self.mipFull.setGeometry(QtCore.QRect(490, 10, 350, 350))
        self.mipFull.setFrameShape(QtGui.QFrame.StyledPanel)
        self.mipFull.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mipFull.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mipFull.setObjectName(_fromUtf8("mipFull"))
        self.colorSpace = QtGui.QGraphicsView(self.centralWidget)
        self.colorSpace.setGeometry(QtCore.QRect(20, 187, 300, 300))
        self.colorSpace.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.colorSpace.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.colorSpace.setObjectName(_fromUtf8("colorSpace"))
        self.colorSpaceLabel = QtGui.QLabel(self.centralWidget)
        self.colorSpaceLabel.setGeometry(QtCore.QRect(30, 157, 281, 20))
        self.colorSpaceLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.colorSpaceLabel.setObjectName(_fromUtf8("colorSpaceLabel"))
        self.intensitySlider = QtGui.QSlider(self.centralWidget)
        self.intensitySlider.setGeometry(QtCore.QRect(360, 187, 22, 290))
        self.intensitySlider.setMaximum(255)
        self.intensitySlider.setPageStep(50)
        self.intensitySlider.setProperty("value", 145)
        self.intensitySlider.setOrientation(QtCore.Qt.Vertical)
        self.intensitySlider.setInvertedAppearance(False)
        self.intensitySlider.setInvertedControls(False)
        self.intensitySlider.setObjectName(_fromUtf8("intensitySlider"))
        self.mipDynamic = QtGui.QGraphicsView(self.centralWidget)
        self.mipDynamic.setGeometry(QtCore.QRect(490, 370, 350, 350))
        self.mipDynamic.setFrameShape(QtGui.QFrame.StyledPanel)
        self.mipDynamic.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mipDynamic.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.mipDynamic.setObjectName(_fromUtf8("mipDynamic"))
        self.volumeSelect = QtGui.QComboBox(self.centralWidget)
        self.volumeSelect.setGeometry(QtCore.QRect(310, 97, 51, 26))
        self.volumeSelect.setMaxVisibleItems(10)
        self.volumeSelect.setSizeAdjustPolicy(QtGui.QComboBox.AdjustToContentsOnFirstShow)
        self.volumeSelect.setObjectName(_fromUtf8("volumeSelect"))
        self.volumeSelect.addItem(_fromUtf8(""))
        self.volumeSelect.addItem(_fromUtf8(""))
        self.deleteVolume = QtGui.QPushButton(self.centralWidget)
        self.deleteVolume.setGeometry(QtCore.QRect(360, 100, 22, 26))
        self.deleteVolume.setObjectName(_fromUtf8("deleteVolume"))
        self.debugLabel = QtGui.QLabel(self.centralWidget)
        self.debugLabel.setGeometry(QtCore.QRect(30, 510, 401, 161))
        self.debugLabel.setFrameShape(QtGui.QFrame.Box)
        self.debugLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.debugLabel.setObjectName(_fromUtf8("debugLabel"))
        self.volumeLabel = QtGui.QLabel(self.centralWidget)
        self.volumeLabel.setGeometry(QtCore.QRect(250, 100, 59, 26))
        self.volumeLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.volumeLabel.setObjectName(_fromUtf8("volumeLabel"))
        self.addAreaButton = QtGui.QPushButton(self.centralWidget)
        self.addAreaButton.setGeometry(QtCore.QRect(230, 130, 91, 32))
        self.addAreaButton.setObjectName(_fromUtf8("addAreaButton"))
        self.areaSelectionView = QtGui.QGraphicsView(self.centralWidget)
        self.areaSelectionView.setGeometry(QtCore.QRect(350, 187, 21, 290))
        self.areaSelectionView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.areaSelectionView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.areaSelectionView.setObjectName(_fromUtf8("areaSelectionView"))
        self.delAreaButton = QtGui.QPushButton(self.centralWidget)
        self.delAreaButton.setGeometry(QtCore.QRect(320, 130, 81, 32))
        self.delAreaButton.setObjectName(_fromUtf8("delAreaButton"))
        self.modeMenu = QtGui.QComboBox(self.centralWidget)
        self.modeMenu.setGeometry(QtCore.QRect(30, 100, 104, 26))
        self.modeMenu.setObjectName(_fromUtf8("modeMenu"))
        self.modeMenu.addItem(_fromUtf8(""))
        self.modeMenu.addItem(_fromUtf8(""))
        self.modeMenu.addItem(_fromUtf8(""))
        self.plusZoom = QtGui.QPushButton(self.centralWidget)
        self.plusZoom.setGeometry(QtCore.QRect(860, 70, 20, 20))
        self.plusZoom.setObjectName(_fromUtf8("plusZoom"))
        self.minusZoom = QtGui.QPushButton(self.centralWidget)
        self.minusZoom.setGeometry(QtCore.QRect(860, 100, 20, 20))
        self.minusZoom.setObjectName(_fromUtf8("minusZoom"))
        self.importImageButton = QtGui.QPushButton(self.centralWidget)
        self.importImageButton.setGeometry(QtCore.QRect(20, 20, 161, 51))
        self.importImageButton.setObjectName(_fromUtf8("importImageButton"))
        self.intensityLabel = QtGui.QLabel(self.centralWidget)
        self.intensityLabel.setGeometry(QtCore.QRect(340, 160, 141, 16))
        self.intensityLabel.setObjectName(_fromUtf8("intensityLabel"))
        self.saveDynamicButton = QtGui.QPushButton(self.centralWidget)
        self.saveDynamicButton.setGeometry(QtCore.QRect(170, 30, 141, 32))
        self.saveDynamicButton.setObjectName(_fromUtf8("saveDynamicButton"))
        self.applyStackButton = QtGui.QPushButton(self.centralWidget)
        self.applyStackButton.setGeometry(QtCore.QRect(300, 30, 121, 32))
        self.applyStackButton.setObjectName(_fromUtf8("applyStackButton"))
        self.editImageButton = QtGui.QToolButton(self.centralWidget)
        self.editImageButton.setGeometry(QtCore.QRect(860, 10, 26, 22))
        self.editImageButton.setIconSize(QtCore.QSize(16, 16))
        self.editImageButton.setObjectName(_fromUtf8("editImageButton"))
        self.gpuLabel = QtGui.QLabel(self.centralWidget)
        self.gpuLabel.setGeometry(QtCore.QRect(850, 720, 151, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.gpuLabel.setFont(font)
        self.gpuLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.gpuLabel.setObjectName(_fromUtf8("gpuLabel"))
        self.areaSelectionView.raise_()
        self.mipFull.raise_()
        self.colorSpace.raise_()
        self.colorSpaceLabel.raise_()
        self.intensitySlider.raise_()
        self.mipDynamic.raise_()
        self.volumeSelect.raise_()
        self.deleteVolume.raise_()
        self.debugLabel.raise_()
        self.volumeLabel.raise_()
        self.addAreaButton.raise_()
        self.delAreaButton.raise_()
        self.modeMenu.raise_()
        self.plusZoom.raise_()
        self.minusZoom.raise_()
        self.importImageButton.raise_()
        self.intensityLabel.raise_()
        self.saveDynamicButton.raise_()
        self.applyStackButton.raise_()
        self.editImageButton.raise_()
        self.gpuLabel.raise_()
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1005, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        MainWindow.setMenuBar(self.menuBar)
        self.actionMenu = QtGui.QAction(MainWindow)
        self.actionMenu.setObjectName(_fromUtf8("actionMenu"))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.colorSpaceLabel.setText(_translate("MainWindow", "Color Space Selector", None))
        self.volumeSelect.setItemText(0, _translate("MainWindow", "1", None))
        self.volumeSelect.setItemText(1, _translate("MainWindow", "Add Volume...", None))
        self.deleteVolume.setText(_translate("MainWindow", "-", None))
        self.debugLabel.setText(_translate("MainWindow", "DebugView", None))
        self.volumeLabel.setText(_translate("MainWindow", "Colors:", None))
        self.addAreaButton.setText(_translate("MainWindow", "Add Area", None))
        self.delAreaButton.setText(_translate("MainWindow", "Del Area", None))
        self.modeMenu.setItemText(0, _translate("MainWindow", "Manual", None))
        self.modeMenu.setItemText(1, _translate("MainWindow", "Auto", None))
        self.modeMenu.setItemText(2, _translate("MainWindow", "Circular", None))
        self.plusZoom.setText(_translate("MainWindow", "+", None))
        self.minusZoom.setText(_translate("MainWindow", "-", None))
        self.importImageButton.setText(_translate("MainWindow", "Import New Image", None))
        self.intensityLabel.setText(_translate("MainWindow", "Intensity:", None))
        self.saveDynamicButton.setText(_translate("MainWindow", "Save View as Tif", None))
        self.applyStackButton.setText(_translate("MainWindow", "Apply to Stack", None))
        self.editImageButton.setText(_translate("MainWindow", "...", None))
        self.gpuLabel.setText(_translate("MainWindow", "GPU: OFF", None))
        self.actionMenu.setText(_translate("MainWindow", "Menu", None))

