import cv2
import numpy as np
from csv import writer
import noisereduce as nr
import threading
import struct
# import tensorflow.keras as keras
# import librosa
import math
import pyaudio
import wave
import struct
import matplotlib.pyplot as plt
import pyqtgraph as pg
# import PyQt5
from PyQt5 import QtGui, QtWidgets, QtTest
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QTimer, QDateTime, QTime
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow
import time
from SoilGUI import Ui_MainWindow
from datetime import datetime
import time
import RPi.GPIO as GPIO
from PCA9685 import PCA9685
import sys


def process_audio(audio):
        # print(type(audio))
        # decode = np.frombuffer(audio, np.uint32)
        left = audio[0::2]
        right = audio[1::2]
        # l = left.astype(float)
        # r = right.astype(float)
        #left_samples = np.nan_to_num(left_samples, posinf = 3.4e+38, neginf =-3.4e+38)
        #right_samples = np.nan_to_num(right_samples, posinf = 3.4e+38, neginf =-3.4e+38)
        # caudio = l
        # caudio = caudio.astype(np.int32)
        # caudio = caudio.tobytes()
        return left, right

class VideoCapture:

    def __init__(self, video_source=0, width=None, height=None, auto_focus=None,
                 focus=None, exposure=None, auto_exposure=None):
        self.cap = cv2.VideoCapture(video_source)
        self.lock = threading.Lock()
        self.stop = False
        self.ret1, self.frame = self.cap.read()
        self.ret2, self.background = self.cap.read()
        self.annotationFlag = True
        self.startTime = datetime.now()
        self.currentTime = datetime.now()
        self.timestamp1 = self.startTime.strftime("%d:%m:%Y %H:%M:%S")
        self.timestamp2 = self.startTime.strftime("%H:%M:%S")
        self.motionFlag = "Init"
        self.datalogf = self.timestamp1 + ".csv"
        datalog = open(self.datalogf, "w+")
        datalog.write("Time,Motion Detected,Sound Detected\n")
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

    def update_background(self):
        self.ret, self.background = self.get_frame
        
    def update_annotation(self):
        if self.annotationFlag == True:
            self.annotationFlag = False
        else: self.annotationFalg = True
        
    def mark_recording_csv(self,file_name):
        with open(file_name, 'a+', newline='') as self.write_obj:
            csv_writer = writer(self.write_obj)
            csv_writer.writerow("\n")
            csv_writer.writerow("recording started")
            csv_writer.writerow("\n")
    
    def add_data_to_csv(self,file_name, motionFlag):
        self.currentTime = datetime.now()
        date = self.currentTime.strftime("%d:%m:%Y %H:%M:%S")
        if self.motionFlag == "True":
            self.motionStr = "YES"
        elif self.motionFlag == "False":
            self.motionStr = "NO"
        else: self.motionStr = "INVALID"
        with open(file_name, 'a+', newline='') as self.write_obj:
            csv_writer = writer(self.write_obj)
            csv_writer.writerow([date, self.motionStr])
    
    
    def detect_motion(self, annotationFlag):
        ret1, frame = self.ret1, self.frame
        ret2, background = self.ret2, self.background
        self.diff = cv2.absdiff(frame, background)
        self.gray = cv2.cvtColor(self.diff, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(self.gray, (5,5), 0)
        self._, self.thresh = cv2.threshold(self.blur, 10, 250, cv2.THRESH_BINARY)
        self.dilated = cv2.dilate(self.thresh, None, iterations=2)
        self.contours, self._ = cv2.findContours(self.dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for self.contour in self.contours:
            (x, y, w, h) = cv2.boundingRect(self.contour)
            
            if annotationFlag == True and cv2.contourArea(self.contour) < 100 and cv2.contourArea(self.contour) > 25:
                cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)
            
            if cv2.contourArea(self.contour) < 1:
                return frame, "True"
            else:
                return frame, "False"
            
        
    def process_frame(self):
        with self.lock:
            self.ret, self.frame = self.get_frame()
            framePro, self.motionFlag = self.detect_motion(self.annotationFlag)
            self.currentTime = datetime.now()
            self.checkTime = self.currentTime.strftime("%d:%m:%Y %H:%M:%S")
            if self.checkTime != self.timestamp1:
                self.add_data_to_csv(self.datalogf, self.motionFlag)
                #cv2.imwrite(self.timestamp1 + '.jpg', self.frame)
                self.timestamp1 = self.checkTime
            self.frame = self.background
            self.ret2, self.background = self.get_frame()
            return framePro

class AudioCapture:

    def __init__(self, FORMAT = pyaudio.paInt32, CHANNELS = 2, RATE = 44100, INPUT = True,
                 CHUNK = 1050, INDEX = 0):
        self.format = FORMAT
        self.channels = CHANNELS
        self.rate = RATE
        self.input = INPUT
        self.chunk = CHUNK
        self.index = INDEX
        self.audio_filename = 'audio-%s.wav' % time.strftime("%Y-%m-%d-%H-%M-%S")

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=self.input,
                                  frames_per_buffer=self.chunk, input_device_index=self.index)
        self.lock = threading.Lock()
        self.stop = False
        self.stream.start_stream()
        if self.stream.is_active():
           print("Stream is active")
        self.frames = []
        self.lframes = []
        self.rframes = []
        if not self.stream.is_active():
            raise ValueError("Unable to open audio source", INDEX)


    def get_audio(self):
        with self.lock:
            full_wave = self.stream.read(num_frames=self.chunk, exception_on_overflow=False)
            return full_wave

    def get_split_audio(self):
        with self.lock:
            audio = self.stream.read(num_frames=self.chunk, exception_on_overflow=False)
            left = audio[0::2]
            right = audio[1::2]
            return left, right

    def save_audio(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        left = "left" + self.audio_filename
        waveFile = wave.open(left, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.p.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.lframes))
        waveFile.close()

        right = "right" + self.audio_filename
        waveFile = wave.open(right, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.p.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.rframes))
        waveFile.close()


class Deployment:

    def __init__(self, filters=None, video=None):
        if filters is None:
            filters = []
        if video is None:
            video = VideoCapture()
        audio  = AudioCapture()
        self.filters = filters
        self.video = video
        self.audio = audio
        self.audio.frames = []
        self.out = None


    def save_img(self, file_save, image):
        cv2.imwrite(file_save, image)
        print(file_save + " was saved")
        return True

    def record_video(self):
        frame = self.video.process_frame()
        self.out.write(frame)


    def record_audio(self):
        left, right = self.audio.get_split_audio()
        self.audio.lframes.append(left)
        self.audio.rframes.append(right)
        return left, right


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QTimer(self)
        self.audio_timer = QTimer(self)
        self.wd_timer = QTimer(self)
        self.wd_timer.setInterval(2000)
        
        self.video_source = 0
        #self.video_source = "vid-2021-03-01-18-41-05.avi"
        
        
        self.pwm = PCA9685()
        self.rot_x = 12
        self.rot_y = 56
        self.pwm.setPWMFreq(50)
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
        self.ui.Start.clicked.connect(self.start_record)
        self.ui.Stop.clicked.connect(self.stop_camera)
        self.ui.Up.clicked.connect(self.tilt_up)
        self.ui.Down.clicked.connect(self.tilt_down)
        self.ui.Left.clicked.connect(self.tilt_left)
        self.ui.Right.clicked.connect(self.tilt_right)
        self.ui.positive.clicked.connect(self.savePositive)
        self.ui.negative.clicked.connect(self.saveNegative)
        self.start_camera()
        self.wd_timer.start()
        self.wd_timer.timeout.connect(self.stop_servo)
        self.ui.anotation.clicked.connect(self.deployment.video.update_annotation)

    def get_data_update_plot(self):
        # reads audio input from microphone
        left, right = self.deployment.audio.get_split_audio()
        left = np.frombuffer(left, dtype=np.int32)
        right = np.frombuffer(right, dtype=np.int32)
        self.ldata_line.setData(self.chunk_range, left)
        self.rdata_line.setData(self.chunk_range, right)

    def get_frame_update_plot(self):
        # reads audio input from microphone
        left = self.deployment.audio.lframes[-1]
        right = self.deployment.audio.rframes[-1]
        left = np.frombuffer(left, dtype=np.int32)
        right = np.frombuffer(right, dtype=np.int32)
        self.ldata_line.setData(self.chunk_range, left)
        self.rdata_line.setData(self.chunk_range, right)
        
    def stop_servo(self):
        self.wd_timer.stop()
        print("wd triggered")
        self.pwm.exit_PCA9685()
        self.wd_timer.start()
        
    def tilt_up(self):
        self.wd_timer.start()
        self.pwm.start_PCA9685()
        self.rot_y = self.rot_y - 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        self.pwm.setRotationAngle(0, self.rot_y)
        self.deployment.video.update_background
        #self.pwm.exit_PCA9685()

    def tilt_down(self):
        self.wd_timer.start()
        self.pwm.start_PCA9685()
        self.rot_y = self.rot_y + 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        self.pwm.setRotationAngle(0, self.rot_y)
        self.deployment.video.update_background
        #self.pwm.exit_PCA9685()

    def tilt_right(self):
        self.wd_timer.start()
        self.pwm.start_PCA9685()
        self.rot_x = self.rot_x - 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        self.pwm.setRotationAngle(1, self.rot_x)
        self.deployment.video.update_background
        #self.pwm.exit_PCA9685()

    def tilt_left(self):
        self.wd_timer.start()
        self.pwm.start_PCA9685()
        self.rot_x = self.rot_x + 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        self.pwm.setRotationAngle(1, self.rot_x)
        self.deployment.video.update_background
        #self.pwm.exit_PCA9685()

    def start_camera(self):
        print('Started Camera')
        # self.stop_camera()
        self.deployment = Deployment(filters=self.filter_run,
                                     video=VideoCapture(video_source=self.video_source, width=self.width,
                                                        height=self.height, auto_focus=self.auto_focus,
                                                        focus=self.focus,
                                                        exposure=self.exposure, auto_exposure=self.auto_exposure))
        self.timer.setInterval(10)
        self.chunk_range = np.arange(0, self.deployment.audio.chunk*2)

        self.timer.timeout.connect(self.update)
        self.timer.start()
        interval = 1000 * (1 / self.deployment.audio.rate)
        self.audio_timer.setInterval(interval)
        left, right = self.deployment.audio.get_split_audio()
        lgraph_data = np.frombuffer(left, dtype=np.int32)
        rgraph_data = np.frombuffer(right, dtype=np.int32)
        self.ldata_line = self.ui.lgraph.plot(self.chunk_range, lgraph_data)
        self.rdata_line = self.ui.rgraph.plot(self.chunk_range, rgraph_data)
        # self.ui.graph.clear()
        self.audio_timer.timeout.connect(self.get_data_update_plot)
        self.audio_timer.start()

    def start_record(self):
        print('Started Recording')
        self.deployment.video.mark_recording_csv(self.deployment.video.datalogf)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.deployment.out = cv2.VideoWriter('vid-%s.avi' % time.strftime("%Y-%m-%d-%H-%M-%S"), fourcc, 20.0, (640, 480))
        self.timer.timeout.connect(self.deployment.record_video)
        self.audio_timer.timeout.disconnect(self.get_data_update_plot)
        self.audio_timer.timeout.connect(self.deployment.record_audio)
        self.audio_timer.timeout.connect(self.get_frame_update_plot)

    def stop_camera(self):
        self.deployment.audio.save_audio()
        try:
            self.timer.timeout.disconnect(self.deployment.record_video)
            self.audio_timer.timeout.disconnect(self.deployment.record_audio)
            # place to disconect recorder
        except Exception:
            pass
        QtTest.QTest.qWait(100)

        print("Recording Stopped")

    def update(self):
        frame = self.deployment.video.process_frame()
        self.displayImage(frame)

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
        frame = self.deployment.video.process_frame()
        self.file_name = 'img-%s.jpg' % time.strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = "/negative/"
        self.file_save = self.file_path + self.file_name
        self.deployment.save_img(self.file_save, frame)
        print(self.file_save + " has been saved to Negative")
    def savePositive(self):
        frame = self.deployment.video.process_frame()
        self.file_name = 'img-%s.jpg' % time.strftime("%Y-%m-%d-%H-%M-%S")
        self.file_path = "/positive/"
        self.file_save = self.file_path + self.file_name
        self.deployment.save_img(self.file_save, frame)
        print(self.file_save + " has been saved to Positive")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())