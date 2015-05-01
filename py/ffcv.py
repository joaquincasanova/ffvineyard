import numpy as np
import cv2
import matplotlib.pyplot as plt

def nothing(x):
    pass

def normalize(x):
    y=(x-np.min(x))/(np.max(x)-np.min(x))*255
    return y

def channelops(mat, n, sigC, sigD):
    
    mat=np.float32(mat)
    mat = cv2.bilateralFilter(mat, n, sigC, sigD)
    mu = cv2.blur(mat,(n,n))
    mdiff=mu-mat
    mat2=cv2.blur(np.float64(mdiff*mdiff),(n,n))
    sd = np.float32(cv2.sqrt(mat2))
    matn=normalize(mat)
    mun=normalize(mu)
    sdn=normalize(sd)
    
    matr=mat.reshape((np.size(matn)))
    mur=mu.reshape((np.size(mun)))
    sdr=sd.reshape((np.size(sdn)))

    features=np.transpose(np.array([matr, mur, sdr]))
    return features

def readsplit(imname):
        
    img = cv2.imread(imname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    b,g,r = cv2.split(img)
    ndvi = (r-b)/(r+b)
    return h, s, v, img
    
def bilat_adjust(mat):

    bilat = cv2.bilateralFilter(mat, 7, 7, 7)

    cv2.namedWindow('bilat')
    cv2.createTrackbar('n','bilat',0,21,nothing)
    cv2.createTrackbar('sigC','bilat',0,1000,nothing)
    cv2.createTrackbar('sigD','bilat',0,21,nothing)
    while(1):
        cv2.imshow('bilat',bilat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        n = cv2.getTrackbarPos('n','bilat')
        sigC = cv2.getTrackbarPos('sigC','bilat')
        sigD = cv2.getTrackbarPos('sigD','bilat')
        bilat = cv2.bilateralFilter(mat, n, sigC, sigD)


    cv2.destroyAllWindows()

    return bilat, n, sigC, sigD
    
def canny_contours(mat, n):
    edges = cv2.Canny(mat,25,50,n)

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
            
        edges = cv2.Canny(mat,minval,maxval,n)

    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return contours, minval, maxval

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
   
    l=(B==255)*0+(G==255)*1+(R==255)*2+(np.logical_and(np.logical_and(B==0, G==0),R==0))*3
    labels=np.float32(l.reshape(np.size(B))[:,np.newaxis])
    
    return labels

def labels_to_rgb_2(labels,rows,cols):
    
    R=np.uint8((labels==0)*255)
    G=np.uint8((labels==1)*255)
    B=np.uint8(np.zeros(G.shape))
    B=B.reshape((rows,cols))
    G=G.reshape((rows,cols))
    R=R.reshape((rows,cols))
    
    segment=np.uint8(cv2.merge((B,G,R)))

    return segment

def rgb_to_labels_2(segment):
    B,G,R = cv2.split(segment)
   
    l=(G<255)*0+(G==255)*1
    labels=np.float32(l.reshape(np.size(G))[:,np.newaxis])
    
    return labels

cv2.destroyAllWindows()
labelsname = "../images/al0.jpg" 
trainname = "../images/at0.jpg" 
testname = "../images/a0.jpg" 

c1,c2,c3,img=readsplit(trainname)
bilat, n, sigC, sigD = bilat_adjust(c1)
contours, minval, maxval = canny_contours(c1, n)

train = channelops(c1, n, sigC, sigD)
train = np.hstack((train,channelops(c3, n, sigC, sigD)))

segment = cv2.imread(labelsname)
labels = rgb_to_labels_2(segment)

c1,c2,c3,img=readsplit(testname)

test = channelops(c1, n, sigC, sigD)
test = np.hstack((test,channelops(c3, n, sigC, sigD)))

rows, cols = c1.shape
cv2.namedWindow('image')
cv2.imshow('image',img)
cv2.waitKey(1)    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

      
knn = cv2.KNearest()
em = cv2.EM(2,cv2.EM_COV_MAT_DIAGONAL)
nb = cv2.NormalBayesClassifier()    
svm_params = dict( kernel_type = cv2.SVM_RBF,
                    svm_type = cv2.SVM_C_SVC)
nb.train(train,labels)
knn.train(train,labels)

varIdx=None
samplesIdx=None
svm = cv2.SVM()
svm.train_auto(train,labels,varIdx,samplesIdx, params=svm_params)
svm.save('svm_data.dat')

result = svm.predict_all(test)
segment=labels_to_rgb_2(result,rows,cols)
cv2.namedWindow("segment svm")
cv2.imshow("segment svm",segment)
cv2.waitKey(1)

knnk = 25
ret, result,neighbours,dist = knn.find_nearest(test,k=knnk)
segment=labels_to_rgb_2(result,rows,cols)
cv2.namedWindow("segment knn")
cv2.imshow("segment knn",segment)
cv2.waitKey(1)

ret, result=nb.predict(test)
segment=labels_to_rgb_2(result,rows,cols)
cv2.namedWindow('segments nb')
cv2.imshow('segments nb',segment)
cv2.waitKey(1)

ret, ll, result, probs = em.train(test)
segment=labels_to_rgb_2(result,rows,cols)
cv2.namedWindow('segments em')
cv2.imshow('segments em',segment)
cv2.waitKey(1)

ret, result, centers = cv2.kmeans(test, 2, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
segment=labels_to_rgb_2(result,rows,cols)
cv2.namedWindow('segments kmeans')
cv2.imshow('segments kmeans',segment)
cv2.waitKey(0)    
cv2.destroyAllWindows()
#
