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
    
    features=np.transpose(np.array([mat, mu, sd]))
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

def em_test(features):
    em = cv2.EM(4,cv2.EM_COV_MAT_DIAGONAL)
    ret, ll, labels, probs = em.train(features)

    return labels

def labels_to_rgb(labels,rows,cols)
    B=(labels==0)*255
    G=(labels==1)*255
    R=(labels==2)*255
    B=B.reshape((rows,cols))
    G=G.reshape((rows,cols))
    R=R.reshape((rows,cols))
    
    segment=np.uint8(cv2.merge((B,G,R)))
    cv2.imshow('segment',np.uint8(segment))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segment

for imnum in ['a', 'r']:
    imname = "../images/{}0.jpg".format(imnum) 
    c1,c2,c3,img=readsplit(imname)
    n=3

    bilat = cv2.bilateralFilter(c1, n, 5, 5)
    features = channelops(bilat, n)
    #contours = canny_contours(c1, n)
    #cv2.drawContours(img, contours, -1, (255,255,255), 3)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    labels=em_test(features)
    
labelsname = "../images/al0.jpg" 
trainname = "../images/at0.jpg" 
testname = "../images/a0.jpg" 

labels = cv2.imread(labelsname)
l1,l2,l3 = cv2.split(labels)
l1.copyTo(l)
l(l1==255)=0
l(l2==255)=1
l(l3==255)=2

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

c1,c2,c3,img=readsplit(trainname)
n=7

bilat = cv2.bilateralFilter(c1, n, 5, 5)
features = channelops(bilat, n)

bilat = cv2.bilateralFilter(c3, n, 5, 5)
features = np.hstack((features,channelops(bilat, n)))

rows, cols = features.shape
l=l.reshape((rows,1))

svm = cv2.SVM()
svm.train(features,l, params=svm_params)
svm.save('svm_data.dat')


c1,c2,c3,img=readsplit(testname)
n=7

bilat = cv2.bilateralFilter(c1, n, 5, 5)
features = channelops(bilat, n)

bilat = cv2.bilateralFilter(c3, n, 5, 5)
features = np.hstack((features,channelops(bilat, n)))

result = svm.predict_all(features)

    
