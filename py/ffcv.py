import numpy as np
import cv2

for imnum in range(1,6):

    imname = "../images/a{}.jpg".format(imnum) 

    img = cv2.imread(imname)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blur = cv2.bilateralFilter(img, 5,7,7)
    cv2.imshow('image',blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    cv2.imshow('image',hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
