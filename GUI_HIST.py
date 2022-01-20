import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
# from rec_web import *
import cv2
import numpy as np
import pickle


def build_squares(img):
    x, y, w, h = 420, 100, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if np.any(imgCrop == None):
                imgCrop = img[y:y + h, x:x + w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y + h, x:x + w]))
            # print(imgCrop.shape)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x += w + d
        if np.any(crop == None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop


class MainWindow(QDialog):

    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('OpenCv.ui', self)
        self.image = None
        self.flagPressedC, self.flagPressedS = False, False
        self.imgCrop = None
        self.finalhistImg = None
        self.start_button_1.setEnabled(True)
        self.adjust_button_1.setEnabled(False)
        self.save_button_1.setEnabled(False)
        self.start_button_1.clicked.connect(self.start_webcam1)
        self.adjust_button_1.clicked.connect(self.start_webcam2)
        self.save_button_1.clicked.connect(self.save_hist)

    def start_webcam1(self):
        # Capture video from external camera
        self.capture = cv2.VideoCapture(1)
        if self.capture.read()[0] == False:
            # if external camera not found, use inbuilt camera
            self.capture = cv2.VideoCapture(0)
        x, y, w, h = 300, 100, 300, 300
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame1)
        self.timer.start(5)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame2)
        self.timer.start(5)

    def start_webcam2(self, ss):
        # Capture video from external camera
        self.capture = cv2.VideoCapture(1)
        if self.capture.read()[0] == False:
            # if external camera not found, use inbuilt camera
            self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.flagPressedC = True
        self.update_frame3(self.flagPressedC, False)

    def update_frame1(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        if not self.flagPressedS:
            self.imgCrop = build_squares(self.image)
        self.displayImage(self.image,1)

    def update_frame2(self):
        hsvCrop = cv2.cvtColor(self.imgCrop, cv2.COLOR_BGR2HSV)
        self.hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([self.hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        dst1 = dst.copy()
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.thresh = cv2.merge((thresh, thresh, thresh))
        self.displayImage(thresh, 2)


    def update_frame3(self,FPC,FPS):
        if FPC:
            self.displayImage(self.thresh, 3)
        if FPS:
            with open("hist", "wb") as f:
                pickle.dump(self.finalhistImg, f)
            self.stop_webcam()


    def displayImage(self, img, sel):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # [0]=rows, [1]=cols, [2]=channels
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        # BGR to RGB
        outImage = outImage.rgbSwapped()
        if sel == 1:
            self.camera_display.setPixmap(QPixmap.fromImage(outImage))
            self.camera_display.setScaledContents(True)
            self.adjust_button_1.setEnabled(True)
            self.start_button_1.setEnabled(False)

        if sel == 2:
            self.threshhold_display.setPixmap(QPixmap.fromImage(outImage))
            self.threshhold_display.setScaledContents(True)
            self.adjust_button_1.setEnabled(True)
            self.start_button_1.setEnabled(False)

        if sel == 3:
            self.finalhistImg = self.hist
            self.toggle_display.setPixmap(QPixmap.fromImage(outImage))
            self.toggle_display.setScaledContents(True)
            self.adjust_button_1.setEnabled(True)
            self.save_button_1.setEnabled(True)
            self.start_button_1.setEnabled(False)

    def adjust_hist(self):
        # Capture video from external camera
        self.capture = cv2.VideoCapture(1)
        if self.capture.read()[0] == False:
            # if external camera not found, use inbuilt camera
            self.capture = cv2.VideoCapture(0)
        x, y, w, h = 300, 100, 300, 300
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame2)
        self.timer.start(5)

    def save_hist(self):
        self.flagPressedS = True
        self.update_frame3(False, self.flagPressedS)

    def stop_webcam(self):
        self.capture.release()
        self.timer.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('ASL recognizer - Setting up Histogram')
    window.show()
    sys.exit(app.exec_())
