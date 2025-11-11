# Image Gradients

import numpy as np
import cv2 
from matplotlib import pyplot as plt

image_path='./img/crops.png'
img = cv2.imread(image_path, 1)

def wind(image):
    cv2.namedWindow('Crops', cv2.WINDOW_NORMAL)
    cv2.imshow('Crops', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#wind(img)

sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
#wind(sobel_x)

laplacian = cv2.Laplacian(img, cv2.CV_8U)
#wind(laplacian)

# Edge and Feature Detection

edges = cv2.Canny(img, 100, 200)
wind(edges)

# Line Detection

lines = cv2.HoughLines(edges, 1, np.pi/180,200)
lines

if lines is not None:
    for iterator in lines:
        rho = iterator[0][0]
        theta = iterator[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
else:
    print("No lines detected.")

wind(img)

# A Simple GeoComputaion Operation

