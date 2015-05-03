import numpy as np
import cv2

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

def ndvi_calc(imname):
        
    img = cv2.imread(imname)
    b,g,r = cv2.split(img)
    s=np.shape(r)
    uno = np.ones(s)
    nmax = np.maximum(-1*uno,(r-b)/(r+b))
    ndvi = np.minimum(nmax,uno)

    return ndvi, img

def readsplit(imname):
        
    img = cv2.imread(imname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    return h, s, v, img

def opening_adjust(mat):

    kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    cv2.namedWindow('open')
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

    cv2.destroyAllWindows()

    return opening, n

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

def rgb_to_binary_2(segment):
    B,G,R = cv2.split(segment)
   
    labels=(G<255)*0+(G==255)*1
    return labels

def rgb_to_labels_2(segment):
    l=rgb_to_binary_2(segment)
    labels=np.float32(l.reshape(np.size(l))[:,np.newaxis])
    
    return labels


