import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
%matplotlib qt

# input caliberation image
img=mpimg.imread('camera_cal/calibration1.jpg')
plt.imshow(img)

objpoints=[]
imgpoints=[]

objp=np.zeros((6*8,3),np.float32)
objp[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, corners=cv2.findChessboardCorners(gray,(8,6),None)

if ret == True:
    imgpoints.append(corners)
    objpoints.append(objp)
    
img=cv2.drawChessboardCorners(img,(6,8), corners, ret)
plt.imshow(img)