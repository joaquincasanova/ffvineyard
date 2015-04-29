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
    mat2=cv2.blur(np.float64(mat*mat),(n,n))
    mu2=np.float64(mu*mu)
    sd = np.float32(cv2.sqrt((mat2-mu2)))
    mat=normalize(mat)
    mu=normalize(mu)
    sd=normalize(sd)
    
    mat=mat.reshape((np.size(mat)))
    mu=mu.reshape((np.size(mu)))
    sd=sd.reshape((np.size(sd)))
    
    features=np.transpose(np.array([mat, mu]))#, sd]))
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
n=7

train = channelops(c1, n)
train = np.hstack((train,channelops(c3, n)))

segment = cv2.imread(labelsname)
labels = rgb_to_labels(segment)

##svm_params = dict( kernel_type = cv2.SVM_LINEAR,
##                    svm_type = cv2.SVM_C_SVC,
##                    C=1.0,
##                    gamma=5.383 )

#em = cv2.EM(4,cv2.EM_COV_MAT_DIAGONAL)
#ret, ll, labels, probs = em.train(train)

knn = cv2.KNearest()
knn.train(train,labels)
##svm = cv2.SVM()
##svm.train(features,labels, params=svm_params)
##svm.save('svm_data.dat')



c1,c2,c3,img=readsplit(testname)
n=7
rows, cols = c1.shape

cv2.namedWindow('image')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

test = channelops(c1, n)
test = np.hstack((test,channelops(c3, n)))

ret,result,neighbours,dist = knn.find_nearest(test,k=4)
#result = svm.predict_all(features)
#ret, result = em.predict(test)

segment=labels_to_rgb(result,rows,cols)
cv2.namedWindow('segments')
cv2.imshow('segments',segment)
cv2.waitKey(0)
cv2.destroyAllWindows()

    
#

    
