import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os
import sqlite3


def pickle_images_labels():
    images_labels = []
    images = glob("gestures/*/*.jpg")
    images.sort()
    for image in images:
        print(image)
        label = image[image.find(os.sep) + 1: image.rfind(os.sep)]
        img = cv2.imread(image, 0)
        images_labels.append((np.array(img, dtype=np.uint8), int(label)))
    return images_labels


images_labels = pickle_images_labels()
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
images, labels = zip(*images_labels)
print("Length of images_labels", len(images_labels))

train_images = images[:int(4 / 5 * len(images))]
print("Length of train_images", len(train_images))
with open("train_images", "wb") as f:
    pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(4 / 5 * len(labels))]
print("Length of train_labels", len(train_labels))
with open("train_labels", "wb") as f:
    pickle.dump(train_labels, f)
del train_labels

test_images = images[int(4 / 5 * len(images)):int(9 / 10 * len(images))]
print("Length of test_images", len(test_images))
with open("test_images", "wb") as f:
    pickle.dump(test_images, f)
del test_images

test_labels = labels[int(4 / 5 * len(labels)):int(9 / 10 * len(images))]
print("Length of test_labels", len(test_labels))
with open("test_labels", "wb") as f:
    pickle.dump(test_labels, f)
del test_labels

val_images = images[int(9 / 10 * len(images)):]
print("Length of val_images", len(val_images))
with open("val_images", "wb") as f:
    pickle.dump(val_images, f)
del val_images

val_labels = labels[int(9 / 10 * len(labels)):]
print("Length of val_labels", len(val_labels))
with open("val_labels", "wb") as f:
    pickle.dump(val_labels, f)
del val_labels


# storing g_id and g_name of all gestures into gesture_db.db
def store_in_db(ges_id, ges_name):
    connection = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (ges_id, ges_name)
    connection.execute(cmd)
    connection.commit()


if not os.path.exists("gesture_db.db"):
    conn = sqlite3.connect("gesture_db.db")
    create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT " \
                       "NOT NULL ) "
    conn.execute(create_table_cmd)
    conn.commit()

    for i in range(0, 26):
        j = chr(ord('@') + (i + 1))
        store_in_db(i, j)

    # for i in range(26, 36):
    #     store_in_db(i, (i - 26))
