import cv2
import numpy as np
import os
def testwrite(img, pfx, imnum, i, j):

    if i==None:
        if imnum < 100:
            oname = "../images/out/{}00790{}.JPG".format(pfx,imnum) 
        else:
            oname = "../images/out/{}0079{}.JPG".format(pfx,imnum) 
        retval=cv2.imwrite(oname,img)
        print "Wrote ", oname, retval        
    else:   
        if imnum < 100:
            oname = "../images/out/{}00790{}_{}_{}.JPG".format(pfx,imnum,i,j) 
        else:
            oname = "../images/out/{}0079{}_{}_{}.JPG".format(pfx,imnum,i,j) 
        retval=cv2.imwrite(oname,img)
        print "Wrote ", oname, retval

os.system("rm ../images/out/*.JPG")
imnum=34
imnum_max=264
if imnum < imnum_max:
    imname = "../images/G00790{}.JPG".format(imnum) 
else:
    imname = "../images/G0079{}.JPG".format(imnum)
print "Img ", imname
img = cv2.imread(imname)
while imnum<=264:    
    if img==None:
        break

    img = cv2.pyrDown(img)
    img = cv2.pyrDown(img)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    h,s,v = cv2.split(hsv)
    l,a,b = cv2.split(lab) 
    a=cv2.equalizeHist(a) 
    #h=cv2.equalizeHist(h)
    testwrite(a, "A", imnum, None, None)
    testwrite(h, "H", imnum, None, None)
  
    sigD=2
    T=121
    t=27
    n=2
    
    ksize=(4*sigD+1,4*sigD+1)
    ab = a
    ab = cv2.adaptiveBilateralFilter(a, ksize, sigD)
    retval,at = cv2.threshold(ab,t,1,cv2.THRESH_BINARY_INV)
    lt=67
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*n+1,2*n+1))
    ao = cv2.morphologyEx(at, cv2.MORPH_OPEN, kernel)
    testwrite(ab, "AB", imnum, None, None)
    testwrite(at*255, "AT", imnum, None, None)
    
    contours, hierarchy = cv2.findContours(ao,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    nn=np.size(contours)
    area = np.zeros([nn,1])
    cx = np.zeros([nn,1])
    cy = np.zeros([nn,1])
    loc = np.zeros([nn,1])

    
    for i in range(0,nn,1):
        M = cv2.moments(contours[i])
        if M['m00']==0:
            cx[i] = None
            cy[i] = None
        else:
            cx[i] = int(M['m10']/M['m00'])
            cy[i] = int(M['m01']/M['m00']) 
        
        if (cx[i]>xcols*0.20 and cx[i]<xcols*0.8):
            area[i] = cv2.contourArea(contours[i])
            cv2.drawContours(img, contours, i, (0,0,0), 3)
        else:
            pass

    area_sort = np.sort(area)

    loc = np.argsort(area) 

    testwrite(img, "AC", imnum, None, None)

    yrows, xcols = ao.shape
    xsize = 75
    ysize = 80 
    tp = xsize*ysize
    I = xcols/xsize
    J = yrows/ysize
    lg, tg = 0, 0
    for i in range(0,I,1):
        for j in range(0,J,1): 
            ao_ij = ao[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]
            gp = tp-np.sum(ao_ij)
            fg = (tp)/(gp)
            if fg>.75:
                lg=lg+gp
            tg = tg+gp
    TP = yrows*xcols
    ff = 1-tg/TP
    fc = 1-lg/TP
    leaf = np.sqrt(area[nn-2])
    hd = 12*0.3048
    view = 50*np.pi/180
    wv = hd*np.tan(view/2)
    b=np.shape(ao)
    px = np.sqrt(b[1]*b[0])
    m_per_pix = wv/px
    leaf = leaf*m_per_pix
    print ff, fc, leaf
    img = None
    
    while img == None and imnum<imnum_max:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
    
