import sys
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
from rec_web import *
import cv2
import numpy as np
import pickle


class MainWindow(QDialog):

    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.setStyleSheet("QWidget {background-image: url(bckg.jpg)}")
        self.hist_btn.setStyleSheet('QPushButton {background-color: red; color: black;}')
        self.recognize_btn.setStyleSheet('QPushButton {background-color: red; color: black;}')
        self.font = QFont()
        self.font.setFamily("Arial")
        self.font.setPointSize(20)
        self.title.setStyleSheet("QLineEdit { color: #FFFFFF;}")
        self.title.setFont(self.font)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('ASL recognizer - For Deaf and Dumb')
    window.show()
    sys.exit(app.exec_())
