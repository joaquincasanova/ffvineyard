import numpy as np
import cv2
import matplotlib.pyplot as plt

def normalize(x):
    y=(x-np.min(x))/(np.max(x)-np.min(x))*255
    return y

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
    mu=normalize(mu)
    sd=normalize(sd)
    grad=normalize(grad)
    false = cv2.merge((mu,sd,grad))
    cv2.imshow('false',np.uint8(false))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    mu=mu.reshape((np.size(mu)))
    sd=sd.reshape((np.size(sd)))
    grad=grad.reshape((np.size(grad)))
    features=np.transpose(np.array([mu, sd]))
    print features.shape
    return features

def readsplit(imname):
        
    img = cv2.imread(imname)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    b,g,r = cv2.split(img)
    
    return h, s, v
    
for imnum in ['a', 'r']:
    imname = "../images/{}0.jpg".format(imnum) 
    h,s,v=readsplit(imname)
    n=3
    features = channelops(h, n)
    features = np.hstack((features,channelops(v, n)))
    for n in [13]:
        features = np.hstack((features,channelops(h, n)))
        features = np.hstack((features,channelops(v, n)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(features, 4, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(h.shape)
    B=(labels==0)*255
    G=(labels==1)*255
    R=(labels==2)*255
    segment=np.uint8(cv2.merge((B,G,R)))
    cv2.imshow('segment',np.uint8(segment))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    em = cv2.EM(4,cv2.EM_COV_MAT_DIAGONAL)
    ret, ll, labels, probs = em.train(features)
    labels = labels.reshape(h.shape)
    B=(labels==0)*255
    G=(labels==1)*255
    R=(labels==2)*255
    segment=np.uint8(cv2.merge((B,G,R)))
    cv2.imshow('segment',np.uint8(segment))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
