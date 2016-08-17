# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'correctionwindow.ui'
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

class Ui_CorrectionWindow(object):
    def setupUi(self, CorrectionWindow):
        CorrectionWindow.setObjectName(_fromUtf8("CorrectionWindow"))
        CorrectionWindow.resize(949, 547)
        self.centralWidget = QtGui.QWidget(CorrectionWindow)
        self.centralWidget.setObjectName(_fromUtf8("centralWidget"))
        self.beforeView = QtGui.QGraphicsView(self.centralWidget)
        self.beforeView.setGeometry(QtCore.QRect(20, 90, 400, 400))
        self.beforeView.setFrameShape(QtGui.QFrame.NoFrame)
        self.beforeView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.beforeView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.beforeView.setObjectName(_fromUtf8("beforeView"))
        self.afterView = QtGui.QGraphicsView(self.centralWidget)
        self.afterView.setGeometry(QtCore.QRect(500, 90, 400, 400))
        self.afterView.setFrameShape(QtGui.QFrame.NoFrame)
        self.afterView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.afterView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.afterView.setObjectName(_fromUtf8("afterView"))
        self.beforeLabel = QtGui.QLabel(self.centralWidget)
        self.beforeLabel.setGeometry(QtCore.QRect(18, 70, 401, 20))
        self.beforeLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.beforeLabel.setObjectName(_fromUtf8("beforeLabel"))
        self.afterLabel = QtGui.QLabel(self.centralWidget)
        self.afterLabel.setGeometry(QtCore.QRect(500, 70, 401, 20))
        self.afterLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.afterLabel.setObjectName(_fromUtf8("afterLabel"))
        self.importButton = QtGui.QPushButton(self.centralWidget)
        self.importButton.setGeometry(QtCore.QRect(20, 30, 161, 41))
        self.importButton.setObjectName(_fromUtf8("importButton"))
        self.layerSlider = QtGui.QSlider(self.centralWidget)
        self.layerSlider.setGeometry(QtCore.QRect(450, 90, 26, 391))
        self.layerSlider.setMaximum(0)
        self.layerSlider.setOrientation(QtCore.Qt.Vertical)
        self.layerSlider.setObjectName(_fromUtf8("layerSlider"))
        self.layerLabel = QtGui.QLabel(self.centralWidget)
        self.layerLabel.setGeometry(QtCore.QRect(430, 60, 91, 16))
        self.layerLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layerLabel.setObjectName(_fromUtf8("layerLabel"))
        self.thresholdMode = QtGui.QCheckBox(self.centralWidget)
        self.thresholdMode.setGeometry(QtCore.QRect(260, 10, 221, 41))
        self.thresholdMode.setObjectName(_fromUtf8("thresholdMode"))
        self.alignButton = QtGui.QPushButton(self.centralWidget)
        self.alignButton.setGeometry(QtCore.QRect(840, 30, 91, 32))
        self.alignButton.setAutoDefault(False)
        self.alignButton.setDefault(False)
        self.alignButton.setObjectName(_fromUtf8("alignButton"))
        self.channelSelectMenu = QtGui.QComboBox(self.centralWidget)
        self.channelSelectMenu.setGeometry(QtCore.QRect(670, 30, 104, 32))
        self.channelSelectMenu.setObjectName(_fromUtf8("channelSelectMenu"))
        self.channelSelectMenu.addItem(_fromUtf8(""))
        self.channelSelectMenu.addItem(_fromUtf8(""))
        self.channelSelectMenu.addItem(_fromUtf8(""))
        self.channelSelectMenu.addItem(_fromUtf8(""))
        self.saveButton = QtGui.QPushButton(self.centralWidget)
        self.saveButton.setGeometry(QtCore.QRect(640, 0, 101, 32))
        self.saveButton.setAutoDefault(False)
        self.saveButton.setDefault(False)
        self.saveButton.setFlat(False)
        self.saveButton.setObjectName(_fromUtf8("saveButton"))
        self.saveStackButton = QtGui.QPushButton(self.centralWidget)
        self.saveStackButton.setGeometry(QtCore.QRect(840, 0, 91, 32))
        self.saveStackButton.setObjectName(_fromUtf8("saveStackButton"))
        self.saveMIPButton = QtGui.QPushButton(self.centralWidget)
        self.saveMIPButton.setGeometry(QtCore.QRect(740, 0, 91, 32))
        self.saveMIPButton.setObjectName(_fromUtf8("saveMIPButton"))
        self.alignField = QtGui.QLineEdit(self.centralWidget)
        self.alignField.setGeometry(QtCore.QRect(790, 30, 41, 21))
        self.alignField.setObjectName(_fromUtf8("alignField"))
        self.thresholdMenu = QtGui.QComboBox(self.centralWidget)
        self.thresholdMenu.setGeometry(QtCore.QRect(200, 20, 61, 26))
        self.thresholdMenu.setObjectName(_fromUtf8("thresholdMenu"))
        self.thresholdMenu.addItem(_fromUtf8(""))
        self.thresholdMenu.addItem(_fromUtf8(""))
        self.thresholdMenu.addItem(_fromUtf8(""))
        self.thresholdMenu.addItem(_fromUtf8(""))
        self.thresholdMenu.addItem(_fromUtf8(""))
        self.thresholdMenu.addItem(_fromUtf8(""))
        CorrectionWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtGui.QMenuBar(CorrectionWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 949, 22))
        self.menuBar.setObjectName(_fromUtf8("menuBar"))
        CorrectionWindow.setMenuBar(self.menuBar)

        self.retranslateUi(CorrectionWindow)
        self.thresholdMenu.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(CorrectionWindow)

    def retranslateUi(self, CorrectionWindow):
        CorrectionWindow.setWindowTitle(_translate("CorrectionWindow", "CorrectionWindow", None))
        self.beforeLabel.setText(_translate("CorrectionWindow", "Before Normalizing", None))
        self.afterLabel.setText(_translate("CorrectionWindow", "After Normalizing", None))
        self.importButton.setText(_translate("CorrectionWindow", "Import Image (.czi)", None))
        self.layerLabel.setText(_translate("CorrectionWindow", "LayerSlider", None))
        self.thresholdMode.setText(_translate("CorrectionWindow", "Apply Threshold in Normalizing", None))
        self.alignButton.setText(_translate("CorrectionWindow", "Align", None))
        self.channelSelectMenu.setItemText(0, _translate("CorrectionWindow", "All", None))
        self.channelSelectMenu.setItemText(1, _translate("CorrectionWindow", "#1: Red", None))
        self.channelSelectMenu.setItemText(2, _translate("CorrectionWindow", "#2: Green", None))
        self.channelSelectMenu.setItemText(3, _translate("CorrectionWindow", "#3: Blue", None))
        self.saveButton.setText(_translate("CorrectionWindow", "Save Image", None))
        self.saveStackButton.setText(_translate("CorrectionWindow", "Save Stack", None))
        self.saveMIPButton.setText(_translate("CorrectionWindow", "Save MIP", None))
        self.alignField.setText(_translate("CorrectionWindow", "5", None))
        self.thresholdMenu.setItemText(0, _translate("CorrectionWindow", "1", None))
        self.thresholdMenu.setItemText(1, _translate("CorrectionWindow", "2", None))
        self.thresholdMenu.setItemText(2, _translate("CorrectionWindow", "3", None))
        self.thresholdMenu.setItemText(3, _translate("CorrectionWindow", "4", None))
        self.thresholdMenu.setItemText(4, _translate("CorrectionWindow", "5", None))
        self.thresholdMenu.setItemText(5, _translate("CorrectionWindow", "6", None))

