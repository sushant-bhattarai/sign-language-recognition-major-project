import cv2
import os

gest_folder = "gestures"
image_x = 75
image_y = 75
for g_id in os.listdir(gest_folder):
	for i in range(2400):
		path = gest_folder + "/" + g_id + "/" + str(i + 1) + ".jpg"
		new_path = gest_folder + "/" + g_id + "/" + str(i + 1) + ".jpg"
		print(new_path)
		img = cv2.imread(path, 0)
		img = cv2.resize(img, (image_x, image_y))
		cv2.imwrite(new_path, img)