from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys


def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200, 100, 840, 480)
    win.setWindowTitle("ASL Recognition")

    win.show()
    sys.exit(app.exec_())

window()