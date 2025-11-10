import numpy as np
import cv2 
from matplotlib import pyplot as plt

image_path='./img/bicycle.jpg'
img = cv2.imread(image_path, 1)

def wind(image):
    cv2.namedWindow('Bicycle', cv2.WINDOW_NORMAL)
    cv2.imshow('Bicycle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


Circle = cv2.circle(img, (900, 770), 230, (0,0,255), 20)
#wind(img)

#2 Function of mouse to draw circle
def click2circle(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x,y), 50, (0,255,0), 4)

cv2.namedWindow('Click2Circle', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Click2Circle', click2circle)

while True:
    cv2.imshow('Click2Circle', img)
    a=cv2.waitKey(1)
    if a==27:
        break
cv2.destroyAllWindows()

#3 Resizing, cropping and exporting

imgres = cv2.resize(img, dsize=None, fx=0.2,fy=0.2, interpolation=cv2.INTER_CUBIC)

cv2.namedWindow('Resize', cv2.WINDOW_NORMAL)
cv2.imshow('Resize', imgres)
cv2.waitKey(0)
cv2.destroyAllWindows()

cropped=imgres[20:768, 30:770]
wind(cropped)

#cv2.imwrite("wheel.png", cropped)

