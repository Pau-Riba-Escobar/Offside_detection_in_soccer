import numpy as np
import cv2

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

def extract_players(im):
    # hsv
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = 255 - mask
    kernel = np.ones(stride_kernel,np.uint8)
    kernel_aud = np.ones(stride_aud,np.uint8)
    kernel_close = np.ones(stride_close,np.uint8)
    thresh_players = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    thresh_aud = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_aud)
    thresh_aud = cv2.morphologyEx(thresh_aud, cv2.MORPH_OPEN, kernel_close)
    thresh_aud = 255 - thresh_aud
    thresh = cv2.bitwise_and( thresh_aud , thresh_players)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)