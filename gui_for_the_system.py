import sys

import pyttsx3
from PyQt5.QtGui import QFont
# from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi
import cv2
import pickle
import numpy as np
import os
import sqlite3
from threading import Thread
from keras.models import load_model


def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    for i in range(10):
        for j in range(5):
            if np.any(imgCrop is None):
                imgCrop = img[y:y + h, x:x + w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y + h, x:x + w]))
            # print(imgCrop.shape)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            x += w + d
        if np.any(crop is None):
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop))
        imgCrop = None
        x = 420
        y += h + d
    return crop


def get_hand_hist_for_hist_generation():
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    while True:
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        blackboard_for_hist = np.zeros((480, 180, 3), dtype=np.uint8)
        combined = np.hstack((img, blackboard_for_hist))
        # cv2.putText(combined, "COVER ALL GREEN", (660, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        # cv2.putText(combined, "SQUARES WITH HAND", (660, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        # cv2.putText(combined, "AND CLICK C", (660, 90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
        # cv2.putText(combined, "TO CAPTURE", (660, 110), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        cv2.putText(combined, "COVER", (660, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "ALL", (660, 110), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "GREEN", (660, 140), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "SQUARES", (660, 170), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "WITH", (660, 200), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "HAND", (660, 230), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "AND", (660, 260), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "CLICK", (660, 290), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "C", (660, 320), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "TO", (660, 350), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(combined, "CAPTURE", (660, 380), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
            flagPressedC = True
            hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        elif keypress == ord('s'):
            flagPressedS = True
            break
        if flagPressedC:
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cv2.filter2D(dst, -1, disc, dst)
            blur = cv2.GaussianBlur(dst, (11, 11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh, thresh, thresh))
            # cv2.imshow("res", res)
            cv2.imshow("Thresh", thresh)
        if not flagPressedS:
            imgCrop = build_squares(combined)
        # cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Set hand histogram", combined)
    cam.release()
    cv2.destroyAllWindows()
    with open("hist", "wb") as f:
        pickle.dump(hist, f)


def get_num_of_classes():
    return len(os.listdir('gestures/'))


engine = pyttsx3.init()
engine.setProperty('rate', 120)
voices = engine.getProperty('voices')
# print(voices[0])
engine.setProperty('voice', voices[1].id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras.h5')


def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape


image_x, image_y = get_image_size()


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
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]


def split_sentence(text, num_of_words):
    list_words = text.split()
    length = len(list_words)
    splitted_sentence = []
    b_index = 0
    e_index = num_of_words
    while length > 0:
        part = ""
        for word in list_words[b_index:e_index]:
            part = part + " " + word
        splitted_sentence.append(part)
        b_index += num_of_words
        e_index += num_of_words
        length -= num_of_words
    return splitted_sentence


def put_splitted_text_in_blackboard(blackboard, splitted_text):
    y = 200
    for text in splitted_text:
        cv2.putText(blackboard, text, (50, 300), cv2.FONT_HERSHEY_TRIPLEX, 7,
                    (255, 255, 255), 10)
        # cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
        y += 50


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


def recognize_for_gesture():
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    hist = get_hand_hist()
    x, y, w, h = 300, 100, 300, 300
    while True:
        text = ""
        img = cam.read()[1]
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        imgCrop = img[y:y + h, x:x + w]
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y + h, x:x + w]
        (openCV_ver, _, __) = cv2.__version__.split(".")
        if openCV_ver == '3':
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        elif openCV_ver == '4':
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            # print(cv2.contourArea(contour))
            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]

                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                  cv2.BORDER_CONSTANT, (0, 0, 0))

                pred_probab, pred_class = keras_predict(model, save_img)

                if pred_probab * 100 > 80:
                    text = get_pred_text_from_db(pred_class)
                    print(text)
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        splitted_text = split_sentence(text, 2)
        cv2.putText(blackboard, "GESTURE RECOGNITION", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(blackboard, "PREDICTED TEXT: ", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1,
                    (255, 255, 255))
        cv2.putText(blackboard, "PRESS Q TO QUIT", (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
        put_splitted_text_in_blackboard(blackboard, splitted_text)
        # cv2.putText(blackboard, text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("Threshold", thresh)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
            break

    cam.release()
    cv2.destroyAllWindows()


def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1 + h1, x1:x1 + w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                      (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2), cv2.BORDER_CONSTANT,
                                      (0, 0, 0))
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab * 100 > 80:
        text = get_pred_text_from_db(pred_class)
    return text


hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True


def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    cv2.filter2D(dst, -1, disc, dst)
    blur = cv2.GaussianBlur(dst, (11, 11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh, thresh, thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y + h, x:x + w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return img, contours, thresh


def say_text(text):
    if not is_voice_on:
        return
    # while engine._inLoop:
    #     pass
    engine.say(text)
    engine.runAndWait()


def text_mode(cam):
    global is_voice_on
    text = ""
    word = ""
    count_same_frame = 0
    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        old_text = text
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                text = get_pred_from_contour(contour, thresh)
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 15:
                    if len(text) == 1:
                        Thread(target=say_text, args=(text,)).start()
                    word = word + text
                    if word.startswith('I/Me '):
                        word = word.replace('I/Me ', 'I ')
                    elif word.endswith('I/Me '):
                        word = word.replace('I/Me ', 'me ')
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
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "ASL WORDING", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (255, 255, 255))
        cv2.putText(blackboard, "PREDICTED TEXT: ", (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, text, (50, 270), cv2.FONT_HERSHEY_TRIPLEX, 7, (255, 255, 0), 10)
        cv2.putText(blackboard, "PRESS V TO TOGGLE VOICE", (30, 427), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
        cv2.putText(blackboard, "PRESS Q TO QUIT", (30, 457), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
        cv2.putText(blackboard, "WORD: " + word, (30, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        if is_voice_on:
            cv2.putText(blackboard, "VOICE ON", (450, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
        else:
            cv2.putText(blackboard, "VOICE OFF", (450, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)
        cv2.imshow("Threshold", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('q') or keypress == ord('Q'):
            break
        if (keypress == ord('v') or keypress == ord('V')) and is_voice_on:
            is_voice_on = False
        elif (keypress == ord('v') or keypress == ord('V')) and not is_voice_on:
            is_voice_on = True
    cam.release()
    cv2.destroyAllWindows()
    # if keypress == ord('c'):
    #     return 2
    # else:
    #     return 0


def recognize_for_wording():
    cam = cv2.VideoCapture(1)

    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)
    text = ""
    word = ""
    count_same_frame = 0
    text_mode(cam)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('GUI_for_system.ui', self)

        self.font = QFont()
        self.font.setFamily("Arial")
        self.font.setPointSize(35)
        self.hand_hist_button.clicked.connect(self.start_hand_hist)
        self.recognize_gesture_button.clicked.connect(self.start_recognize)
        self.wording_button.clicked.connect(self.start_wording)

    def start_hand_hist(self):
        self.recognize_gesture_button.setEnabled(True)
        self.wording_button.setEnabled(True)
        self.success.setText(" ")
        get_hand_hist_for_hist_generation()
        self.success.setText("Hand histogram set successfully!")
        self.recognize_gesture_button.setEnabled(True)
        self.wording_button.setEnabled(True)
        self.hand_hist_button.setEnabled(True)

    def start_recognize(self):
        self.hand_hist_button.setEnabled(True)
        self.wording_button.setEnabled(True)
        self.success.setText(" ")

        recognize_for_gesture()
        self.success.setText("You pressed Q to quit!")
        self.recognize_gesture_button.setEnabled(True)
        self.wording_button.setEnabled(True)
        self.hand_hist_button.setEnabled(True)

    def start_wording(self):
        self.hand_hist_button.setEnabled(True)
        self.recognize_gesture_button.setEnabled(True)
        self.success.setText(" ")

        recognize_for_wording()
        self.success.setText("You pressed Q to quit!")
        self.recognize_gesture_button.setEnabled(True)
        self.wording_button.setEnabled(True)
        self.hand_hist_button.setEnabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle('ASL recognition using CNN')
    window.show()
    sys.exit(app.exec_())
