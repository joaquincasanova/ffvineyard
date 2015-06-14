import cv2
import numpy as np
import os
    
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
    cv2.createTrackbar('n','edges',0,255,nothing)
    cv2.createTrackbar('MinVal','edges',0,255,nothing)
    cv2.createTrackbar('MaxVal','edges',0,255,nothing)
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

os.system("rm ../images/*A*.JPG")
os.system("rm ../images/*H*.JPG")
os.system("rm ../images/*GC*.JPG")

imnum=34
imnum_max=136
if imnum < imnum_max:
    imname = "../images/G00790{}.JPG".format(imnum) 
else:
    imname = "../images/G0079{}.JPG".format(imnum)
print "Img ", imname
img = cv2.imread(imname)
while imnum<=264:    
    if img==None:
        break

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    h,s,v = cv2.split(hsv)
    l,a,b = cv2.split(lab) 
    a=cv2.equalizeHist(a) 
    h=cv2.equalizeHist(h)
    testwrite(a, "A", imnum, None, None)
    testwrite(h, "H", imnum, None, None)
        
    if imnum==34:
        ab, sigC, sigD = bilat_adjust(a)
        af, T = fast_adjust(ab) 
        ae, n, minval, maxval = canny_adjust(ab)
        testwrite(ab, "AB", imnum, None, None)
        testwrite(af, "AF", imnum, None, None)
        testwrite(ae, "AE", imnum, None, None)
    else:
        ab = cv2.bilateralFilter(a, -1, sigC, sigD)
        ae = cv2.Canny(ab,minval,maxval,n)
        fast = cv2.FastFeatureDetector(T)
        kp = fast.detect(ab,None)
        af = cv2.drawKeypoints(ab, kp, color=(255,0,0))
        testwrite(ab, "AB", imnum, None, None)
        testwrite(ae, "AE", imnum, None, None)
        testwrite(af, "AF", imnum, None, None)

    img = None
    
    while img == None and imnum<imnum_max:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
    
