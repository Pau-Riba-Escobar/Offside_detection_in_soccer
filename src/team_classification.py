import numpy as np
import cv2
from matplotlib import pyplot as plt

# Descripors based on RGB mean
def getMeanDescriptors(im, contours):
    descriptors = []
    for player in contours:
        x,y,w,h = cv2.boundingRect(player)
        row_mean = np.mean(im[y:y+h,x:x+w,:], axis=1)
        overall_mean = np.round(np.mean(row_mean, axis=0)).astype("uint8")
        descriptors.append(overall_mean)
    return np.array(descriptors)

# Descriptor of all the RGB values pixels in the rect
def getRectDescriptors(im, contours):
    descriptors = []
    for player in contours:
        x,y,w,h = cv2.boundingRect(player)
        hsv_rect = cv2.cvtColor(im[y:y+h,x:x+w,:], cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv_rect], [0],mask = None, histSize=[180], ranges=[0,179])
        hist = hist/np.max(hist)
        #zeros_ = np.argwhere(hist == 0)
        descriptors.append(hist)
    return np.array(descriptors)#.reshape(-1,1)
