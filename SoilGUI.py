# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SoilGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1016, 626)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(29, 29, 921, 531))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Frame = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.Frame.setMinimumSize(QtCore.QSize(200, 100))
        self.Frame.setText("")
        self.Frame.setObjectName("Frame")
        self.horizontalLayout.addWidget(self.Frame)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.Right = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Right.setObjectName("Right")
        self.gridLayout.addWidget(self.Right, 1, 2, 1, 1)
        self.Stop = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Stop.setObjectName("Stop")
        self.gridLayout.addWidget(self.Stop, 3, 2, 1, 1)
        self.Left = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Left.setObjectName("Left")
        self.gridLayout.addWidget(self.Left, 1, 0, 1, 1)
        self.Start = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Start.setObjectName("Start")
        self.gridLayout.addWidget(self.Start, 3, 0, 1, 1)
        self.negative = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.negative.setObjectName("negative")
        self.gridLayout.addWidget(self.negative, 4, 2, 1, 1)
        self.positive = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.positive.setObjectName("positive")
        self.gridLayout.addWidget(self.positive, 4, 0, 1, 1)
        self.Up = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Up.setMinimumSize(QtCore.QSize(113, 32))
        self.Up.setObjectName("Up")
        self.gridLayout.addWidget(self.Up, 0, 1, 1, 1)
        self.Down = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.Down.setObjectName("Down")
        self.gridLayout.addWidget(self.Down, 2, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Right.setText(_translate("MainWindow", "Right"))
        self.Stop.setText(_translate("MainWindow", "Stop"))
        self.Left.setText(_translate("MainWindow", "Left"))
        self.Start.setText(_translate("MainWindow", "Start"))
        self.negative.setText(_translate("MainWindow", "Negative"))
        self.positive.setText(_translate("MainWindow", "Positive"))
        self.Up.setText(_translate("MainWindow", "Up"))
        self.Down.setText(_translate("MainWindow", "Down"))
