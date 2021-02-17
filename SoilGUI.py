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
        MainWindow.resize(800, 417)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Up = QtWidgets.QPushButton(self.centralwidget)
        self.Up.setGeometry(QtCore.QRect(580, 130, 113, 32))
        self.Up.setObjectName("Up")
        self.Down = QtWidgets.QPushButton(self.centralwidget)
        self.Down.setGeometry(QtCore.QRect(580, 190, 113, 32))
        self.Down.setObjectName("Down")
        self.Left = QtWidgets.QPushButton(self.centralwidget)
        self.Left.setGeometry(QtCore.QRect(480, 160, 113, 32))
        self.Left.setObjectName("Left")
        self.Right = QtWidgets.QPushButton(self.centralwidget)
        self.Right.setGeometry(QtCore.QRect(680, 160, 113, 32))
        self.Right.setObjectName("Right")
        self.Start = QtWidgets.QPushButton(self.centralwidget)
        self.Start.setGeometry(QtCore.QRect(500, 320, 113, 32))
        self.Start.setObjectName("Start")
        self.Stop = QtWidgets.QPushButton(self.centralwidget)
        self.Stop.setGeometry(QtCore.QRect(640, 320, 113, 32))
        self.Stop.setObjectName("Stop")
        self.Frame = QtWidgets.QLabel(self.centralwidget)
        self.Frame.setGeometry(QtCore.QRect(49, 45, 401, 301))
        self.Frame.setText("")
        self.Frame.setObjectName("Frame")
        self.positive = QtWidgets.QPushButton(self.centralwidget)
        self.positive.setGeometry(QtCore.QRect(500, 360, 113, 32))
        self.positive.setObjectName("positive")
        self.negative = QtWidgets.QPushButton(self.centralwidget)
        self.negative.setGeometry(QtCore.QRect(640, 360, 113, 32))
        self.negative.setObjectName("negative")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Up.setText(_translate("MainWindow", "Up"))
        self.Down.setText(_translate("MainWindow", "Down"))
        self.Left.setText(_translate("MainWindow", "Left"))
        self.Right.setText(_translate("MainWindow", "Right"))
        self.Start.setText(_translate("MainWindow", "Start"))
        self.Stop.setText(_translate("MainWindow", "Stop"))
        self.positive.setText(_translate("MainWindow", "Positive"))
        self.negative.setText(_translate("MainWindow", "Negative"))

