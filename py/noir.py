#import cv2
#import numpy as np

from ffcv import *

cv2.destroyAllWindows()
noirname = "../images/noir_1.bmp" 
bluename = "../images/noir_blue_1.bmp" 

ndvi,img=ndvi_calc(noirname)
cv2.namedWindow('image')
cv2.imshow('image',img)
cv2.waitKey(1)    

ndvin = normalize(ndvi)
cv2.namedWindow('ndvi')
cv2.imshow('ndvi',np.uint8(ndvin))
cv2.waitKey(1)    

ndvi,img=ndvi_calc(bluename)
cv2.namedWindow('image b')
cv2.imshow('image b',img)
cv2.waitKey(1)    

ndvin = normalize(ndvi)
cv2.namedWindow('ndvi b')
cv2.imshow('ndvi b',np.uint8(ndvin))
cv2.waitKey(0)    

cv2.destroyAllWindows()
cv2.destroyAllWindows()
labelsname = "../images/al0.jpg" 
trainname = "../images/at0.jpg" 
testname = "../images/a0.jpg" 

c1,c2,c3,img=readsplit(testname)
##bilat, n, sigC, sigD = bilat_adjust(c1)
n=7
sigC=190
sigD=1
test = channelops(c1, n, sigC, sigD)
test = np.hstack((test,channelops(c3, n, sigC, sigD)))
rows, cols = c1.shape
cv2.namedWindow('image')
cv2.imshow('image',img)
cv2.waitKey(1)    

c1,c2,c3,img=readsplit(trainname)
train = channelops(c1, n, sigC, sigD)
train = np.hstack((train,channelops(c3, n, sigC, sigD)))

segment = cv2.imread(labelsname)
labels = rgb_to_labels_2(segment)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
      
knn = cv2.KNearest()
em = cv2.EM(2,cv2.EM_COV_MAT_DIAGONAL)
em2 = cv2.EM(4,cv2.EM_COV_MAT_DIAGONAL)

##nb = cv2.NormalBayesClassifier()    
##svm_params = dict( kernel_type = cv2.SVM_RBF,
##                    svm_type = cv2.SVM_C_SVC)
##nb.train(train,labels)
knn.train(train,labels)

##varIdx=None
##samplesIdx=None
##svm = cv2.SVM()
##svm.train_auto(train,labels,varIdx,samplesIdx, params=svm_params)
##svm.save('svm_data.dat')
##
##result = svm.predict_all(test)
##segment=labels_to_rgb_2(result,rows,cols)
##cv2.namedWindow("segment svm")
##cv2.imshow("segment svm",segment)
##cv2.waitKey(1)
##
knnk = 25
ret, result,neighbours,dist = knn.find_nearest(test,k=knnk)
segment=labels_to_rgb_2(result,rows,cols)
cv2.namedWindow("segment knn")
cv2.imshow("segment knn",segment)
cv2.waitKey(1)
##
##ret, result=nb.predict(test)
##segment=labels_to_rgb_2(result,rows,cols)
##cv2.namedWindow('segments nb')
##cv2.imshow('segments nb',segment)
##cv2.waitKey(1)

##ret, ll, result, probs = em.train(test)
##segment=labels_to_rgb_2(result,rows,cols)
##cv2.namedWindow('segments em')
##cv2.imshow('segments em',segment)
##cv2.waitKey(1)

##ret, ll, result, probs = em2.train(test)
##segment=labels_to_rgb(result,rows,cols)
##cv2.namedWindow('segments em 4')
##cv2.imshow('segments em 4',segment)
##cv2.waitKey(1)

##ret, result, centers = cv2.kmeans(test, 2, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
##segment=labels_to_rgb_2(result,rows,cols)
##cv2.namedWindow('segments kmeans')
##cv2.imshow('segments kmeans',segment)
##cv2.waitKey(0)    
##cv2.destroyAllWindows()
#
binary = rgb_to_binary_2(segment)
opening, no = opening_adjust(np.uint8(binary*255))
