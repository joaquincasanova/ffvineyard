#import cv2
#import numpy as np

import os

from ffcv import *

def testwrite(img, pfx, imnum, i, j):

    if i==None:
        if imnum < 100:
            oname = "../images/{}00790{}.JPG".format(pfx,imnum) 
        else:
            oname = "../images/{}0079{}.JPG".format(pfx,imnum) 
        retval=cv2.imwrite(oname,img)
        print "Wrote ", oname, retval        
    else:   
        if imnum < 100:
            oname = "../images/{}00790{}_{}_{}.JPG".format(pfx,imnum,i,j) 
        else:
            oname = "../images/{}0079{}_{}_{}.JPG".format(pfx,imnum,i,j) 
        retval=cv2.imwrite(oname,img)
        print "Wrote ", oname, retval

os.system("rm ../images/*_*_*.JPG")

imnum=34
if imnum < 100:
    imname = "../images/G00790{}.JPG".format(imnum) 
else:
    imname = "../images/G0079{}.JPG".format(imnum)
print "Img ", imname
img = cv2.imread(imname)
while imnum<=264:
        
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
            
    h_eq = cv2.equalizeHist(h)
    testwrite(h_eq, "HQ", imnum, None, None)

    edges, contours, minval, maxval=canny_contours(bilat, 25)
    testwrite(edges, "E", imnum, None, None)
    
##    yrows, xcols = gray.shape
##    xsize = 1000
##    ysize = 750 
##    print yrows, xcols
##    I = xcols/xsize
##    J = yrows/ysize
##    for i in range(0,I,1):
##        for j in range(0,J,1): 
##            iimg = img[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize,:]
##            gray_ij = gray[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]
##            h_ij = h[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]
##            h_eq = cv2.equalizeHist(h_ij)
##            thresh = cv2.adaptiveThreshold(gray_ij,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,0)
##            #testwrite(iimg, "G", imnum, i, j)
##            #testwrite(gray_ij, "GR", imnum, i, j)
##            #testwrite(h_ij, "H", imnum, i, j)
##            #testwrite(h_eq, "HQ", imnum, i, j)
##            #testwrite(thresh, "AT", imnum, i, j)
##            h_ij = None
##            gray_ij = None
##            thresh = None
##
##
##    h = None
##    gray = None
##    img = None

 
    while img == None and imnum<265:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
