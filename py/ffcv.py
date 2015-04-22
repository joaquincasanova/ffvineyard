import numpy as np
import cv2

for imnum in range(0,4):

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
##    cv2.imshow('image',hist)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    heq = cv2.equalizeHist(h)
##    cv2.imshow('image',heq)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
    histnorm = np.uint8(normalize(hist, 0, 255))
    cv2.imshow('image',histnorm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##
##    H = np.linspace(0, 180, 180)
##    V = np.linspace(0, 256, 256)
##    VV,HH = np.meshgrid(V,H)
##    print hist.shape
##    ax.plot_surface(HH,VV,hist)
##    plt.show()
##    cv2.waitKey(0)
##    plt.clf()
##    cv2.destroyAllWindows()
##    edges = cv2.Canny(heq,25,50)
##    cv2.imshow('edges',edges)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    
