import numpy as np
import cv2
import matplotlib.pyplot as plt

def channelops(mat, n):
    
    mat=np.float32(mat)
    mu = cv2.blur(mat,(n,n))
    mat2=cv2.blur(np.float64(mat*mat),(n,n))
    mu2=np.float64(mu*mu)
    sd = np.float32(cv2.sqrt((mat2-mu2)))
    ddx = cv2.Sobel(mat,cv2.CV_32F,1,0,n) 
    ddy = cv2.Sobel(mat,cv2.CV_32F,0,1,n)
    grad2=np.float64(ddx*ddx+ddy*ddy)
    grad=np.float32(cv2.sqrt(grad2))
    mu   =     (mu-np.min(mu))/(np.max(mu)-np.min(mu))*255
    sd   =     (sd-np.min(sd))/(np.max(sd)-np.min(sd))*255
    grad = (grad-np.min(grad)(np.max(grad)-np.min(grad))*255
    #false = cv2.merge((mu,sd,grad))
    thresh = cv2.adaptiveThreshold(np.uint8(mu),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7)
##    cv2.imshow('image',np.uint8(mat))
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    cv2.imshow('mean',np.uint8(mu))
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    cv2.imshow('sd',np.uint8(sd))
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    cv2.imshow('grad',np.uint8(grad))
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
    #cv2.imshow('false',np.uint8(false))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imshow('thresh',np.uint8(thresh))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return mu, sd, grad

def features(imname, n):
    img = cv2.imread(imname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    b,g,r = cv2.split(img)
    
    mu, sd, grad = channelops(h, n)
    #mu, sd, grad = channelops(s, n) 
    #mu, sd, grad = channelops(v, n) 
    #mu, sd, grad = channelops(b, n)
    #mu, sd, grad = channelops(g, n)
    #mu, sd, grad = channelops(r, n)   
    
for imnum in range(0,1):
    for n in range(5,6):
    #n=5
        imname = "../images/a{}.jpg".format(imnum) 
        features(imname,n)
    #imname = "../images/r{}.jpg".format(imnum) 
    #n = 11
    #features(imname,n)
    
