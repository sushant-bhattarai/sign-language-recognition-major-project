import sys

from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
# from rec_web import *
import numpy as np
import cv2
import pickle
import numpy as np
# import tensorflow as tf
# from cnn_tf import cnn_model_fn
import os
import sqlite3
import pyttsx3
from keras.models import load_model
from threading import Thread


def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape


def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img


def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


image_x, image_y = get_image_size()
engine = pyttsx3.init("dummy")
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras.h5')


def say_text(text):
    if not is_voice_on:
        return
    while engine._inLoop:
        pass
    engine.say(text)
    engine.runAndWait()


x, y, w, h = 300, 100, 300, 300
text = ""
word = ""
count_same_frame = 0
is_voice_on = True


class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('OpenCv3.ui', self)
        self.image = None
        self.thresh = None
        self.imgCrop = None
        self.imgHSV = None
        self.capture = None

        self.font = QFont()
        self.font.setFamily("Arial")
        self.font.setPointSize(35)
        self.text.setFont(self.font)
        self.stop_button.setEnabled(False)
        self.voice_on_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_webcam)
        self.voice_off_button.clicked.connect(self.voice_off)
        self.voice_on_button.clicked.connect(self.voice_on)
        self.stop_button.clicked.connect(self.stop_webcam)

    def start_webcam(self):
        # Capture video from external camera
        self.capture = cv2.VideoCapture(1)
        if not self.capture.read()[0]:
            # if external camera not found, use inbuilt camera
            self.capture = cv2.VideoCapture(0)
        x, y, w, h = 300, 100, 300, 300
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.stop_button.setEnabled(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.image = cv2.resize(self.image, (640, 480))
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        self.imgCrop = self.image[y:y + h, x:x + w]
        self.imgHSV = cv2.cvtColor(self.imgCrop, cv2.COLOR_BGR2HSV)
        hist = get_hand_hist()
        dst = cv2.calcBackProject([self.imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        self.thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        global text, count_same_frame, word
        old_text = text
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                text = ""
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT,
                                                  (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT,
                                                  (0, 0, 0))
                pred_probab, pred_class = keras_predict(model, save_img)
                if pred_probab * 100 > 80:
                    text = get_pred_text_from_db(pred_class)

                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 20:
                    if len(text) == 1:
                        Thread(target=say_text, args=(text,)).start()
                    word = word + text
                    count_same_frame = 0

            elif cv2.contourArea(contour) < 1000:
                if word != '':
                    # print('yolo')
                    # say_text(text)
                    Thread(target=say_text, args=(word,)).start()
                text = ""
                word = ""
        else:
            if word != '':
                # print('yolo1')
                # say_text(text)
                Thread(target=say_text, args=(word,)).start()
            text = ""
            word = ""

        self.text.setText(text)
        self.word.setText(word)

        self.displayImage(self.image)
        self.displayThresh(thresh)

    def displayImage(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR to RGB
        outImage = outImage.rgbSwapped()
        self.camera_display.setPixmap(QPixmap.fromImage(outImage))
        self.camera_display.setScaledContents(True)

    def displayThresh(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR to RGB
        outImage = outImage.rgbSwapped()
        self.threshhold_display.setPixmap(QPixmap.fromImage(outImage))
        self.threshhold_display.setScaledContents(True)

    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()

    def voice_off(self):
        global is_voice_on
        is_voice_on = False
        self.voice_on_button.setEnabled(True)
        self.voice_off_button.setEnabled(False)

    def voice_on(self):
        global is_voice_on
        is_voice_on = True
        self.voice_on_button.setEnabled(False)
        self.voice_off_button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('ASL recognizer - Creating Words')
    window.show()
    sys.exit(app.exec_())