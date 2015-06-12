import cv2
import numpy as np
import os

def normalize(x):
    y=(x-np.min(x))/(np.max(x)-np.min(x))*255
    return y

def localSD(mat, n):
    
    mat=np.float32(mat)
    mu = cv2.blur(mat,(n,n))
    mdiff=mu-mat
    mat2=cv2.blur(np.float64(mdiff*mdiff),(n,n))
    sd = np.float32(cv2.sqrt(mat2))
    sdn=normalize(sd)

    return sdn

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
os.system("rm ../images/*S*.JPG")
os.system("rm ../images/*Q*.JPG")

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
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    h,s,v = cv2.split(hsv)
    l,a,b = cv2.split(hsv)
    a = cv2.bilateralFilter(a, -1, 180, 3)        
    #h_eq = cv2.equalizeHist(h)
    #a_eq = cv2.equalizeHist(a)
    testwrite(h, "H", imnum, None, None) 
    testwrite(s, "S", imnum, None, None) 
    testwrite(v, "V", imnum, None, None) 
    testwrite(l, "L", imnum, None, None) 
    testwrite(a, "A", imnum, None, None) 
    testwrite(b, "B", imnum, None, None)     

    surf=cv2.SURF(8000)
    kp, des = surf.detectAndCompute(a,None)
    surf_im = cv2.drawKeypoints(a,kp,None,(255,0,0),4)
    testwrite(surf_im, "SURF", imnum, None, None)
    #sdn = localSD(gray, 101)
    #testwrite(sdn, "SDN", imnum, None, None)
##    
##    yrows, xcols = gray.shape
##    xsize = 1000
##    ysize = 750 
##    print yrows, xcols
##    I = xcols/xsize
##    J = yrows/ysize
##    for i in range(0,I,1):
##        for j in range(0,J,1): 
##            img_ij = img[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize,:]
##            h_ij = h[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]
##            gray_ij = gray[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]
##            h_eq = cv2.equalizeHist(h_ij)
##            testwrite(gray_ij, "G", imnum, i, j)
##            testwrite(h_eq, "HQ", imnum, i, j)
##            surf=cv2.SURF(8000)
##            kp, des = surf.detectAndCompute(h_eq,None)
##            surf_ij = cv2.drawKeypoints(h_eq,kp,None,(255,0,0),4)
##            testwrite(surf_ij, "S", imnum, i, j)
##            sdn = localSD(gray_ij, 101)
##            testwrite(sdn, "SDN", imnum, i, j)
##            
##            h_ij = None
##            img_ij = None
##            gray_ij = None
##            kp = None
##            des = None

    img = None
    
    while img == None and imnum<264:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
