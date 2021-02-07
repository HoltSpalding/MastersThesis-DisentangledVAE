import cv2
import os

for subdir, dirs, files in os.walk("../../data/bricks-812/"):
    count = 0
    for file in files:
        filepath = subdir + os.sep + file
        img = cv2.imread(filepath)
        cv2.imwrite("../../usup/grayscale-us/" + str(count) + ".png", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        count += 1