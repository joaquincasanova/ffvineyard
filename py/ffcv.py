import numpy as np
import cv2
import matplotlib.pyplot as plt

for imnum in range(0,3):

    imname = "../images/a{}.jpg".format(imnum) 

    img = cv2.imread(imname)
    blur = cv2.bilateralFilter(img, 5,7,7)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    cv2.imshow('image',h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    h = np.reshape(h,np.size(h))
    v = np.reshape(v,np.size(v))

    plt.hist2d(h, v,[180,256])
    plt.show()
