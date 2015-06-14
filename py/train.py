import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

global img
global pointnum
pointnum = 0
global X
global drawing
drawing = False # true if mouse is pressed
global ix,iy
ix, iy = -1,-1
global rect

# mouse callback functions
def draw_rectangle(event,x,y,flags,param):

    global img, rect
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)
        rect = (ix,iy,x,y)
        
def draw_circle(event,x,y,flags,param):
    global img
    global X, pointnum

    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),2)
        if pointnum==0:
            X = (x,y)
        else:
            np.vstack((X,(x,y)))
        print pointnum, x, y
        pointnum=pointnum+1

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

os.system("rm ../images/*H*.JPG")
os.system("rm ../images/*S*.JPG")
os.system("rm ../images/*V*.JPG")
os.system("rm ../images/*L*.JPG")
os.system("rm ../images/*A*.JPG")
os.system("rm ../images/*B*.JPG")
os.system("rm ../images/*Q*.JPG")
os.system("rm ../images/*T*.JPG")

imnum=34
imnum_max=136
if imnum < imnum_max:
    imname = "../images/G00790{}.JPG".format(imnum) 
else:
    imname = "../images/G0079{}.JPG".format(imnum)
print "Img ", imname

img = cv2.imread(imname)

while imnum<=264:

    Img = img.copy()
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    yrows, xcols = gray.shape
#    rect = (1000, 0, 2000, 2250)

    cv2.namedWindow('image{}'.format(imnum),cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image{}'.format(imnum),draw_circle)

    while(1):
        cv2.imshow('image{}'.format(imnum),img)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()    
    
    pointnum = 0
    X = np.empty([1,2])            
    img = None
    
    while img == None and imnum<imnum_max:
        imnum=imnum+1
        if imnum < 100:
            imname = "../images/G00790{}.JPG".format(imnum) 
        else:
            imname = "../images/G0079{}.JPG".format(imnum)
        print "Img ", imname
        img = cv2.imread(imname)
    
