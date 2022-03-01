import os
import cv2
import pickle
import numpy as np
from glob import glob

gest_folder = "gestures_2"
image_x = 75
image_y = 75
x, y, w, h = 300, 100, 300, 300


def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist


hist = get_hand_hist()

images_labels = []
images = glob("gestures/*/*.jpg")
images.sort()


for g_id in os.listdir(gest_folder):
    for i in range(3000):
        path = gest_folder + "/" + g_id + "/" + chr(ord('@') + (0 + 1)) + i + ".jpg"
        # path = gest_folder + "/A/A1722.jpg"
        print(path)
        img = cv2.imread(path, 1)
        # img = cv2.resize(img, (640, 480))
        # # imgCrop = img[y:y + h, x:x + w]
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        new_image = thresh
        label = path[path.find(os.sep) + 1: path.rfind(os.sep)]
        images_labels.append((np.array(new_image, dtype=np.uint8), int(label)))





