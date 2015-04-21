import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    h,s,v = cv2.split(hsv)
    cv2.imshow('image',h)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('image',v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hist = cv2.calcHist( [hsv], [0, 2], None, [180, 256], [0, 180, 0, 256] )
    cv2.imshow('hist',hist)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    H = np.linspace(0, 180, 180)
    V = np.linspace(0, 256, 256)
    VV,HH = np.meshgrid(V,H)
    print hist.shape
    ax.plot_surface(HH,VV,hist)
    plt.show()
    cv2.waitKey(0)
    plt.clf()
    cv2.destroyAllWindows()
    edges = cv2.Canny(h,25,50)
    cv2.imshow('edges',edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
