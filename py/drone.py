#import cv2
#import numpy as np

from ffcv import *

for imnum in range(34, 104, 10):
    
    print "Preprocesssing"
    cv2.destroyAllWindows()
    if imnum < 100:
        imname = "../images/G00790{}.JPG".format(imnum) 
    else:
        imname = "../images/G0079{}.JPG".format(imnum)
    print "Img ", imname 
    cc1,cc2,cc3,img=readsplit(imname)
    yrows, xcols = cc1.shape
    xsize = 1000
    ysize = 750 
    print yrows, xcols
    I = xcols/xsize
    J = yrows/ysize
    for i in range(0,I,1):
        for j in range(0,J,1): 
            iimg = img[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize,:]
            print "Img ", imname
            if imnum < 100:
                oname = "../images/G00790{}_{}_{}.JPG".format(imnum,i,j) 
            else:
                oname = "../images/G0079{}_{}_{}.JPG".format(imnum,i,j) 
            retval=cv2.imwrite(oname,iimg)
            print "Wrote ", oname, retval
            c1 = cc1[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize] 
            c3 = cc3[j*(ysize-1):j*(ysize-1)+ysize,i*(xsize-1):i*(xsize-1)+xsize]

            if imnum < 100:
                oname = "../images/H00790{}_{}_{}.JPG".format(imnum,i,j) 
            else:
                oname = "../images/H0079{}_{}_{}.JPG".format(imnum,i,j) 
            retval=cv2.imwrite(oname,c1)
            print "Wrote ", oname, retval

            if imnum < 100:
                oname = "../images/V00790{}_{}_{}.JPG".format(imnum,i,j) 
            else:
                oname = "../images/V0079{}_{}_{}.JPG".format(imnum,i,j) 
            retval=cv2.imwrite(oname,c3)
            print "Wrote ", oname, retval

            #bilat, sigC, sigD = bilat_adjust(c1)
            n=17
            sigC=180
            sigD=1
            test = channelops(c1, n, sigC, sigD)
            n=17
            sigC=300
            sigD=3
            test = np.hstack((test,channelops(c3, n, sigC, sigD)))
            rows, cols = c1.shape
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            
            em = cv2.EM(5,cv2.EM_COV_MAT_DIAGONAL)
            ret, ll, result, probs = em.train(test)
            segment=labels_to_rgb(result,rows,cols)
            print "EM segment ", imname
            if imnum < 100:
                oname = "../images/EM00790{}_{}_{}.JPG".format(imnum,i,j) 
            else:
                oname = "../images/EM0079{}_{}_{}.JPG".format(imnum,i,j) 
            cv2.imwrite(oname,segment)
            print "Wrote ", oname
                
            segment = None
            result = None
            em = None
            ll = None
            ret = None
            probs = None
            c1 = None
            c3 = None
            test = None
                    
    cc1 = None
    cc2 = None
    cc3 = None
