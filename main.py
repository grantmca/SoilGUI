import cv2
import numpy as np
import PyQt5
from PyQt5 import QtGui, QtWidgets, QtTest
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, QDateTime, QTime
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
import time
from SoilGUI import Ui_MainWindow

import time
# import RPi.GPIO as GPIO
# from PCA9685 import PCA9685

import sys

class VideoCapture:

    def __init__(self, video_source=0, width=None, height=None, auto_focus=None,
                 focus=None, exposure=None, auto_exposure=None):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)



    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return ret, frame
            else:
                return ret, None
        else:
            return False, None


class Deployment:
    '''All filters must take an image (write an assert)
    '''

    def __init__(self, filters=None, video=None):
        if filters is None:
            filters = []
        if video is None:
            video = VideoCapture()

        self.filters = filters
        self.video = video

    def get_frame(self):
        return self.video.get_frame()

    def save_img(self, file_save, image):
        cv2.imwrite(file_save, image)
        print(file_save + " was saved")
        return True


    # def update_filters(self, )

    def process_frame(self):
        start_time = time.time()
        ret, frame = self.get_frame()
        frame = [frame, None]
        for method in self.filters:
            frame = method(image=frame[0], value=frame[1])


        return frame


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QTimer(self)
        self.button_timer = QTimer(self)
        self.video_source = 0
        # self.pwn = PCA9685()
        self.rot_x = 0
        self.rot_y = 0
        # self.pwm.setPWMFreq(50)
        self.filter_run = []
        self.filter_list = []
        self.height = 2048
        self.width = 2048
        self.auto_focus = True
        self.focus = None
        self.auto_exposure = True
        self.exposure = None
        self.blank_image = np.zeros((20, 20, 3), np.uint8)
        self.deployment = None
        self.ui.Start.clicked.connect(self.start_camera)
        self.ui.Stop.clicked.connect(self.stop_camera)
        self.ui.Up.clicked.connect(self.tilt_up)
        self.ui.Down.clicked.connect(self.tilt_down)
        self.ui.Left.clicked.connect(self.tilt_left)
        self.ui.Right.clicked.connect(self.tilt_right)
        self.ui.positive.clicked.connect(self.savePositive)
        self.ui.negative.clicked.connect(self.saveNegative)

    # def __del__(self):

        # if self.deployment.video.cap.isOpened():
        #     self.deployment.video.cap.release()
    def tilt_up(self):

        self.rot_y = self.rot_y + 5
        print(self.rot_y)
        # self.pwm.setRotationAngle(self.rot_x, self.rot_y)

    def tilt_down(self):
        self.rot_y = self.rot_y - 5
        print(self.rot_y)
        # self.pwm.setRotationAngle(self.rot_x, self.rot_y)
    def tilt_right(self):
        self.rot_x = self.rot_x + 5
        print(self.rot_x)
        # self.pwm.setRotationAngle(self.rot_x, self.rot_y)
    def tilt_left(self):
        self.rot_x = self.rot_x - 5
        print(self.rot_x)
        # self.pwm.setRotationAngle(self.rot_x, self.rot_y)

    def start_camera(self):
        print('Started Camera')
        self.stop_camera()
        self.deployment = Deployment(filters=self.filter_run,
                                     video=VideoCapture(video_source=self.video_source, width=self.width,
                                                        height=self.height, auto_focus=self.auto_focus,
                                                        focus=self.focus,
                                                        exposure=self.exposure, auto_exposure=self.auto_exposure))
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def stop_camera(self):
        self.timer.stop()
        # self.__del__()
        QtTest.QTest.qWait(100)
        if self.deployment is not None:
            self.deployment.video.cap.release()
            self.deployment = None


    def update(self):
        frame = self.deployment.process_frame()
        self.displayImage(frame[0])

    def displayImage(self, frame):
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        self.ui.Frame.setPixmap(QtGui.QPixmap.fromImage(outImage))
        self.ui.Frame.setScaledContents(True)

    def saveNegative(self):
        frame = self.deployment.process_frame()
        self.file_name = 'img-%s.jpg' % time.strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = "/Users/grantmcallister/Developer/PycharmProjects/SoilGUI/negative/"
        self.file_save = self.file_path + self.file_name
        self.deployment.save_img(self.file_save, frame[0])
        print(self.file_save + " has been saved to Negative")
    def savePositive(self):
        frame = self.deployment.process_frame()
        self.file_name = 'img-%s.jpg' % time.strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = "/Users/grantmcallister/Developer/PycharmProjects/SoilGUI/positive/"
        self.file_save = self.file_path + self.file_name
        self.deployment.save_img(self.file_save, frame[0])
        print(self.file_save + " has been saved to Positive")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())