import cv2
import numpy as np
import os

def opening_adjust(mat):

    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    cv2.namedWindow('open',cv2.WINDOW_NORMAL)
    opening = cv2.morphologyEx(mat, cv2.MORPH_OPEN, kernel)
    cv2.createTrackbar('n','open',0,50,nothing)
    while(1):
        cv2.imshow('open',opening)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        n = cv2.getTrackbarPos('n','open')
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*n+1,2*n+1))
        opening = cv2.morphologyEx(mat, cv2.MORPH_OPEN, kernel)
        opening = opening/255

    cv2.destroyAllWindows()

    return opening, n

def thresh_adjust_inv(mat):
    retval,thresh = cv2.threshold(mat,127,255,cv2.THRESH_BINARY_INV)
    
    cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('T','thresh',0,255,nothing)
    while(1):
        cv2.imshow('thresh',thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

            # get current positions of four trackbars
        T = cv2.getTrackbarPos('T','thresh')
        retval,thresh = cv2.threshold(mat,T,255,cv2.THRESH_BINARY_INV)    

    retval,thresh = cv2.threshold(mat,T,1,cv2.THRESH_BINARY_INV)    
    cv2.destroyAllWindows()

    return thresh, T

def thresh_adjust(mat):
    retval,thresh = cv2.threshold(mat,127,255,cv2.THRESH_BINARY)
    
    cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('T','thresh',0,255,nothing)
    while(1):
        cv2.imshow('thresh',thresh)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

            # get current positions of four trackbars
        T = cv2.getTrackbarPos('T','thresh')
        retval,thresh = cv2.threshold(mat,T,255,cv2.THRESH_BINARY)    

    retval,thresh = cv2.threshold(mat,T,1,cv2.THRESH_BINARY)    
    cv2.destroyAllWindows()

    return thresh, T

def fast_adjust(mat):
    fast = cv2.FastFeatureDetector(50)
    kp = fast.detect(mat,None)
    fast_im = cv2.drawKeypoints(mat, kp, color=(255,0,0))

    cv2.namedWindow('keypoints',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('T','keypoints',0,255,nothing)
    while(1):
        cv2.imshow('keypoints',fast_im)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

            # get current positions of four trackbars
        T = cv2.getTrackbarPos('T','keypoints')

        fast = cv2.FastFeatureDetector(T)
        kp = fast.detect(mat,None)
        fast_im = cv2.drawKeypoints(mat, kp, color=(255,0,0))

    cv2.destroyAllWindows()

    return fast_im, T
    
def canny_adjust(mat):
    edges = cv2.Canny(mat,25,50,5)

    cv2.namedWindow('edges',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('n','edges',0,25,nothing)
    cv2.createTrackbar('MinVal','edges',0,1000,nothing)
    cv2.createTrackbar('MaxVal','edges',0,1000,nothing)
    while(1):
        cv2.imshow('edges',edges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

            # get current positions of four trackbars
        n = cv2.getTrackbarPos('n','edges')
        minval = cv2.getTrackbarPos('MinVal','edges')
        maxval = cv2.getTrackbarPos('MaxVal','edges')
            
        edges = cv2.Canny(mat,minval,maxval,n)

    cv2.destroyAllWindows()

    return edges, n, minval, maxval

def bilat_adjust(mat):

    bilat = cv2.bilateralFilter(mat, -1, 7, 7)

    cv2.namedWindow('bilat',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('sigC','bilat',0,100,nothing)
    cv2.createTrackbar('sigD','bilat',0,100,nothing)
    while(1):
        cv2.imshow('bilat',bilat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
  
        sigC = cv2.getTrackbarPos('sigC','bilat')
        sigD = cv2.getTrackbarPos('sigD','bilat')
        bilat = cv2.bilateralFilter(mat, -1, sigC, sigD)

    cv2.destroyAllWindows()

    return bilat, sigC, sigD

def abf_adjust(mat):

    bilat = cv2.adaptiveBilateralFilter(mat, (29,29), 7)

    cv2.namedWindow('bilat',cv2.WINDOW_NORMAL)
    cv2.createTrackbar('sigD','bilat',0,30,nothing)
    while(1):
        cv2.imshow('bilat',bilat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        sigD = cv2.getTrackbarPos('sigD','bilat')
        ksize=(4*sigD+1,4*sigD+1)
        bilat = cv2.adaptiveBilateralFilter(mat, ksize, sigD)

    cv2.destroyAllWindows()

    return bilat, sigD

def nothing(x):
    pass

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
    hb = h
    #ab = cv2.adaptiveBilateralFilter(a, ksize, sigD)
    #hb = cv2.adaptiveBilateralFilter(h, ksize, sigD)
    retval,at = cv2.threshold(ab,t,1,cv2.THRESH_BINARY_INV)
    lt=67
    retval,ht = cv2.threshold(hb,lt,255,cv2.THRESH_BINARY)
    hlt = 200
    lines = cv2.HoughLines(ht,1,np.pi/180,hlt)
    while lines==None:
        hlt=hlt-5
        lines = cv2.HoughLines(ht,1,np.pi/180,hlt)
    
    for rho,theta in lines[0]:
        aa = np.cos(theta)
        bb = np.sin(theta)
        x0 = aa*rho
        y0 = bb*rho
        x1 = int(x0 + 1000*(-bb))
        y1 = int(y0 + 1000*(aa))
        x2 = int(x0 - 1000*(-bb))
        y2 = int(y0 - 1000*(aa))

        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2)
    testwrite(ht, "HT", imnum, None, None)
    #fast = cv2.FastFeatureDetector(T)
    #kp = fast.detect(ab,None)
    #af = cv2.drawKeypoints(ab, kp, color=(255,0,0))
    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*n+1,2*n+1))
    #ao = cv2.morphologyEx(at, cv2.MORPH_OPEN, kernel)
    ad = cv2.dilate(at, kernel, iterations = 4)
    ao = cv2.erode(ad, kernel, iterations = 4)
    testwrite(ab, "AB", imnum, None, None)
    testwrite(at*255, "AT", imnum, None, None)
    #testwrite(af, "AF", imnum, None, None)
    #testwrite(ao*255, "AO", imnum, None, None)
    area = np.zeros([3,1])
    loc = np.zeros([3,1])
    cx = np.zeros([3,1])
    cy = np.zeros([3,1])
    
    contours, hierarchy = cv2.findContours(ao,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area[0] = 0.0
    area[1] = 0.0
    area[2] = 0.0
    loc[0] = 0
    loc[1] = 0
    loc[2] = 0
    idx=0
    for c in contours:
        atest = cv2.contourArea(c)
        if atest>area[0]:
            area[0]=atest
            loc[0] = idx
        else:
            if atest>area[1]:
                area[1]=atest
                loc[1] = idx
            else:                
                if atest>area[2]:
                    area[2]=atest
                    loc[2] = idx
                else:
                    pass            
                    
        idx=idx+1
    for i in [0, 1, 2]:
        M = cv2.moments(contours[int(loc[i])])
        cx[i] = int(M['m10']/M['m00'])
        cy[i] = int(M['m01']/M['m00']) 
        #print area[i], cx[i], cy[i]
    cv2.drawContours(img, contours, int(loc[0]), (0,255,0), 3)    
    cv2.drawContours(img, contours, int(loc[1]), (0,0,255), 3)   
    cv2.drawContours(img, contours, int(loc[2]), (255,0,0), 3)
    testwrite(img, "AC", imnum, None, None)

##    yrows, xcols = ao.shape
##    xsize = 75
##    ysize = 80 
##    tp = xsize*ysize
##    I = xcols/xsize
##    J = yrows/ysize
##    lg, tg = 0, 0
##    for i in range(0,I,1):
##        for j in range(0,J,1): 
##            ao_ij = ao[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]
##            gp = tp-np.sum(ao_ij)
##            fg = (tp)/(gp)
##            if fg>.75:
##                lg=lg+gp
##            tg = tg+gp
##    TP = yrows*xcols
##    ff = 1-tg/TP
##    fc = 1-lg/TP
##
##    print ff, fc
    img = None
    
    while img == None and imnum<imnum_max:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
    
