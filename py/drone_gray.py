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
  
    ret1,th1 = cv2.threshold(h_eq,127,255,cv2.THRESH_BINARY)
    ret2,thresh = cv2.threshold(h_eq,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    testwrite(thresh, "OT", imnum, None, None)
    thresh = cv2.adaptiveThreshold(h_eq,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,2)
    testwrite(thresh, "AT", imnum, None, None)
    h = None
    img = None

 
    while img == None and imnum<265:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
