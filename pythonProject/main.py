# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#converting into hsv image
#green range
lower_green = np.array([36,0,0])
upper_green = np.array([86, 255, 255])
#blue range
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])
#Red range
lower_red = np.array([0,31,255])
upper_red = np.array([176,255,255])
#white range
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])


stride_kernel = (25,25)
stride_aud = (30,30)
stride_close = (150,150)

def iou(box1,box2):

    x1 = max(box1[0],box2[0])
    x2 = min(box1[2],box2[2])
    y1 = max(box1[1] ,box2[1])
    y2 = min(box1[3],box2[3])
    if x1 > x2 or y1>y2:
        return -2
    inter = (x2 - x1)*(y2 - y1)
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1])
    fin_area = area1 + area2 - inter   
    iou = inter/fin_area
        
    return iou



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    
    im = cv2.imread('DataSet/Train/4.jpg')
    new_size = (int(im.shape[0]/2),int(im.shape[1]/2))
    # imatge més petita per a un millor processament
    im = cv2.resize(im,new_size)
    cv2.imshow('hola',im)
    cv2.waitKey(0)

    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    
    cv2.imshow('hola',hsv)
    cv2.waitKey(0) 
    
    mask = cv2.inRange(hsv, lower_green, upper_green)

    cv2.imshow('hola',mask)
    cv2.waitKey(0)
    
    mask = 255 - mask

    kernel = np.ones(stride_kernel,np.uint8)
    kernel_aud = np.ones(stride_aud,np.uint8)
    kernel_close = np.ones(stride_close,np.uint8)
    
    thresh_aud = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_aud)
    thresh_aud = cv2.morphologyEx(thresh_aud, cv2.MORPH_OPEN, kernel_close)
    thresh_aud = 255 - thresh_aud
    
    cv2.imshow('hola',thresh_aud)
    cv2.waitKey(0) 
    
    