import numpy as np
import cv2
import matplotlib.pyplot as plt

def nothing(x):
    pass

def normalize(x):
    y=(x-np.min(x))/(np.max(x)-np.min(x))*255
    return y

def channelops(mat, n):
    
    mat=np.float32(mat)
    mu = cv2.blur(mat,(n,n))
    mdiff=mu-mat
    mat2=cv2.blur(np.float64(mdiff*mdiff),(n,n))
    sd = np.float32(cv2.sqrt(mat2))
    mat=normalize(mat)
    mu=normalize(mu)
    sd=normalize(sd)
    
    mat=mat.reshape((np.size(mat)))
    mu=mu.reshape((np.size(mu)))
    sd=sd.reshape((np.size(sd)))
    
    features=np.transpose(np.array([mu, sd]))
    return features

def readsplit(imname):
        
    img = cv2.imread(imname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    b,g,r = cv2.split(img)
    
    return h, s, v, img
    
def canny_contours(c, n):
    edges = cv2.Canny(c,25,50,n)

    cv2.namedWindow('edges')
    cv2.createTrackbar('MinVal','edges',0,255,nothing)
    cv2.createTrackbar('MaxVal','edges',0,255,nothing)
    while(1):
        cv2.imshow('edges',edges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

            # get current positions of four trackbars
        minval = cv2.getTrackbarPos('MinVal','edges')
        maxval = cv2.getTrackbarPos('MaxVal','edges')
            
        edges[:] = cv2.Canny(c,minval,maxval,n)

    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return contours

def labels_to_rgb(labels,rows,cols):
    
    B=(labels==0)*255
    G=(labels==1)*255
    R=(labels==2)*255
    B=B.reshape((rows,cols))
    G=G.reshape((rows,cols))
    R=R.reshape((rows,cols))
    
    segment=np.uint8(cv2.merge((B,G,R)))

    return segment

def rgb_to_labels(segment):
    B,G,R = cv2.split(segment)
   
    l=(B==255)*0
    l=(G==255)*1
    l=(R==255)*2
    l=(np.logical_and(np.logical_and(B==0, G==0),R==0))*3
    labels=np.float32(l.reshape(np.size(B))[:,np.newaxis])
    
    return labels
    
labelsname = "../images/al0.jpg" 
trainname = "../images/at0.jpg" 
testname = "../images/a0.jpg" 

c1,c2,c3,img=readsplit(trainname)
n=13

train = channelops(c1, n)
train = np.hstack((train,channelops(c3, n)))

segment = cv2.imread(labelsname)
labels = rgb_to_labels(segment)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
##svm_params = dict( kernel_type = cv2.SVM_LINEAR,
##                    svm_type = cv2.SVM_C_SVC,
##                    C=1.0,
##                    gamma=5.383 )
#svm = cv2.SVM()
#knn = cv2.KNearest()
em = cv2.EM(4,cv2.EM_COV_MAT_SPHERICAL)
#nb = cv2.NormalBayesClassifier()

#nb.train(train,labels)
#ret, ll, labels, probs = em.train(test)
#knn.train(train,labels)
#svm.train(train,labels, params=svm_params)
#svm.save('svm_data.dat')



c1,c2,c3,img=readsplit(testname)
rows, cols = c1.shape

#cv2.namedWindow('image')
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

test = channelops(c1, n)
test = np.hstack((test,channelops(c3, n)))

#ret, result,neighbours,dist = knn.find_nearest(test,k=4)
#result = svm.predict_all(features)
#ret, result = em.predict(test)
#ret, result=nb.predict(test)

ret, ll, result, probs = em.train(test)
segment=labels_to_rgb(result,rows,cols)
cv2.namedWindow('segments')
cv2.imshow('segments',segment)
cv2.waitKey(0)
cv2.destroyAllWindows()


ret, result, centers = cv2.kmeans(test, 4, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
segment=labels_to_rgb(result,rows,cols)
cv2.namedWindow('segments')
cv2.imshow('segments',segment)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
#

    
