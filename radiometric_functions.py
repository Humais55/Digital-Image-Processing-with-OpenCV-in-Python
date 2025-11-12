# Imamge colorspace and thresholding

import numpy as np
import cv2 
from matplotlib import pyplot as plt

image_path='./img/bicycle.jpg'
img = cv2.imread(image_path, 1)

image_pathl='./img/lena.png'
imgl = cv2.imread(image_pathl, 0)

def wind(image):
    cv2.namedWindow('Bicycle', cv2.WINDOW_NORMAL)
    cv2.imshow('Bicycle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#wind(hsv)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#wind(gray)

r, t = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#wind(t)

# Image Histogram extraction and manipulaion

#wind(imgl)
hist = cv2.calcHist([imgl], [0], None, [256], [0, 256])
plt.hist(imgl.flatten(), bins=256, range=[0, 256])
plt.show()

equ = cv2.equalizeHist(imgl)
res = np.hstack((imgl, equ))

#wind(res)

# Convolution Based Operations

blur = cv2.blur(imgl, (7, 7))
#wind(blur)

blurG = cv2.GaussianBlur(imgl, (11, 11), 0)
reso = np.hstack((imgl, blurG))
#wind(reso)

# K-mean Classification

imgCL = np.float32(img.reshape((-1, 3)))
crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
K = 4
ret, lab, center = cv2.kmeans(imgCL, K, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
resol = center[lab.flatten()]
resolu = resol.reshape((img.shape))
resolution = np.hstack((img, resolu))
wind(resolution)