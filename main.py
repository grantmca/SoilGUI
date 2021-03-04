import cv2
import numpy as np
import tensorflow.keras as keras
import librosa
import math
import pyaudio
import wave
import struct
import matplotlib.pyplot as plt

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

MODEL_DATA_PATH = "model.h5" #loading the CNN model from tensor flow
NUM_OF_SAMPLRS_FOR_MODEL_INPUT = 22050*30 # number of samples for 1 seconds. if longer is need multiple by the amount of seconds

class  _Audio_Class:

    #creates a model object and a mapping for the classifications
    model = None
    _mapping = [  "birds",
        "cars",
        "crickets",
        "footsteps",
        "motor",
        "rain",
        "talking",
        "wind"

    ]
    _instance = None

    def predict(self,MFCC):
        #extact mfcc
        #MFCC = self.preprocess(file_path) # hape of array is (# of segments, # coefficents)

        #convert the 2d mfcc array to an 4d array
        MFCC = MFCC[np.newaxis, ... ,np.newaxis]                            #      (# of samples for prediction, # of segments, # coefficents, # channels )

        #make prediction
        prediction = self.model.predict(MFCC) # out is a 2d array which contain as the class for classification

        '''  # This array contains the probability that this input is one of the classification
        for i in range(9):
            print("{}:{}".format(self._mapping[i],prediction[0][i]))
        '''
        prediction_index = np.argmax(prediction) # this gets the higest value of probability in the output array of the model_selection

        predicted_keyword = self._mapping[prediction_index] # determing the key word for predictions in mappings

        return predicted_keyword, prediction.flatten()

    def preprocess(self, file_path, n_mfcc = 13, n_fft = 2048, hop_length = 512):

        #load audio files
        signal, sr = librosa.load(file_path)

        #ensure consistency of files length
        if len(signal) > NUM_OF_SAMPLRS_FOR_MODEL_INPUT:
            signal = signal[:NUM_OF_SAMPLRS_FOR_MODEL_INPUT]

        #extract MFCC
        MFCC = librosa.feature.mfcc(signal, n_mfcc = n_mfcc, n_fft = n_fft, hop_length = hop_length)

        return MFCC.T

def Audio_Class():
    # ensuring that there is only 1 instance of Audio_Class
    if _Audio_Class._instance is None:
        _Audio_Class._instance = _Audio_Class()
        _Audio_Class.model = keras.models.load_model(MODEL_DATA_PATH)

    return _Audio_Class._instance

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

    def process_frame(self):
        ret, frame = self.get_frame()



        return frame

class AudioCapture:

    def __init__(self, FORMAT = pyaudio.paInt16, CHANNELS = 1, RATE = 22050, INPUT = True,
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
        self.stream.start_stream()
        if self.stream.is_active():
           print("Stream is active")
        self.frames = []
        if not self.stream.is_active():
            raise ValueError("Unable to open audio source", INDEX)

    def process_audio(self):
        hold = None

    def get_audio(self):
        return self.stream.read(self.chunk, False)

    def save_audio(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        waveFile = wave.open(self.audio_filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.p.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.frames))


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
        data = self.audio.get_audio()
        self.audio.frames.append(data)


class App(QMainWindow):
    def __init__(self):
        super(App, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.timer = QTimer(self)
        self.audio_timer = QTimer(self)
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
        self.ui.Start.clicked.connect(self.start_record)
        self.ui.Stop.clicked.connect(self.stop_camera)
        self.ui.Up.clicked.connect(self.tilt_up)
        self.ui.Down.clicked.connect(self.tilt_down)
        self.ui.Left.clicked.connect(self.tilt_left)
        self.ui.Right.clicked.connect(self.tilt_right)
        self.ui.positive.clicked.connect(self.savePositive)
        self.ui.negative.clicked.connect(self.saveNegative)
        self.start_camera()

    def tilt_up(self):
        self.rot_y = self.rot_y - 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        # self.pwm.setRotationAngle(0, self.rot_y)

    def tilt_down(self):
        self.rot_y = self.rot_y + 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        # self.pwm.setRotationAngle(0, self.rot_y)

    def tilt_right(self):
        self.rot_x = self.rot_x - 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        # self.pwm.setRotationAngle(1, self.rot_x)

    def tilt_left(self):
        self.rot_x = self.rot_x + 2
        print("Y: " + str(self.rot_y))
        print("X: " + str(self.rot_x))
        # self.pwm.setRotationAngle(1, self.rot_x)

    def start_camera(self):
        print('Started Camera')
        # self.stop_camera()
        self.deployment = Deployment(filters=self.filter_run,
                                     video=VideoCapture(video_source=self.video_source, width=self.width,
                                                        height=self.height, auto_focus=self.auto_focus,
                                                        focus=self.focus,
                                                        exposure=self.exposure, auto_exposure=self.auto_exposure))
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.audio_timer.start()

    def start_record(self):
        print('Started Recording')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.deployment.out = cv2.VideoWriter('vid-%s.avi' % time.strftime("%Y-%m-%d-%H-%M-%S"), fourcc, 20.0, (640, 480))
        self.timer.timeout.connect(self.deployment.record_video)
        interval = 1000*(1/self.deployment.audio.rate)
        self.audio_timer.setInterval(interval)
        print("Cal Interval" + str(interval))
        print("Actual Interval" + str(self.audio_timer.interval()))
        self.audio_timer.timeout.connect(self.deployment.record_audio)
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