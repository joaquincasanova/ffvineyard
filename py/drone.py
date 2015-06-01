#import cv2
#import numpy as np

from ffcv import *

for imnum in range(34, 261, 1):
    cv2.destroyAllWindows()
    if imnum < 100:
        imname = "../images/G00790{}.JPG".format(imnum) 
    else:
        imname = "../images/G0079{}.JPG".format(imnum) 
    
    c1,c2,c3,img=readsplit(imname)

    cv2.namedWindow('image')
    cv2.imshow('image',img)
    cv2.waitKey(1)    

    cv2.destroyAllWindows()
    #bilat, n, sigC, sigD = bilat_adjust(c1)
    n=7
    sigC=190
    sigD=10
    test = channelops(c1, n, sigC, sigD)
    test = np.hstack((test,channelops(c3, n, sigC, sigD)))
    rows, cols = c1.shape

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    em2 = cv2.EM(4,cv2.EM_COV_MAT_DIAGONAL)
    ret, ll, result, probs = em2.train(test)
    segment=labels_to_rgb(result,rows,cols)
    cv2.namedWindow('segments em 4')
    cv2.imshow('segments em 4',segment)
    cv2.waitKey(1)

    ret, result, centers = cv2.kmeans(test, 4, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    segment=labels_to_rgb_2(result,rows,cols)
    cv2.namedWindow('segments kmeans')
    cv2.imshow('segments kmeans',segment)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()

    #binary = rgb_to_binary_2(segment)
    #opening, no = opening_adjust(np.uint8(binary*255))
