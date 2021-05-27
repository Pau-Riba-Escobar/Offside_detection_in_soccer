import numpy as np
import cv2

#converting into hsv image
#green range
lower_green = np.array([36,0,0])
upper_green = np.array([86, 255, 255])

stride_kernel = (5,5)
stride_aud = (30,30)
stride_close = (40,40)
stride_erode = (15,1)
stride_dilate = (10,10)

def extract_players(im):
    # hsv
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('green_mask',mask)
    cv2.waitKey(0)
    mask = 255 - mask
    kernel = np.ones(stride_kernel,np.uint8)
    kernel_aud = np.ones(stride_aud,np.uint8)
    kernel_close = np.ones(stride_close,np.uint8)
    kernel_erode = np.ones(stride_erode,np.uint8)
    kernel_final_dilate = np.ones(stride_dilate,np.uint8)
    thresh_players = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('players',thresh_players)
    cv2.waitKey(0)
    thresh_aud = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_aud)
    cv2.imshow('aud',thresh_aud)
    cv2.waitKey(0)
    thresh_aud = cv2.morphologyEx(thresh_aud, cv2.MORPH_OPEN, kernel_close)
    cv2.imshow('aud2',thresh_aud)
    cv2.waitKey(0)
    thresh_aud = 255 - thresh_aud
    comb_thresh = cv2.bitwise_and( thresh_aud , thresh_players)
    thresh_no_noise = cv2.morphologyEx(comb_thresh,cv2.MORPH_ERODE,kernel_erode)
    thresh = cv2.morphologyEx(thresh_no_noise,cv2.MORPH_DILATE,kernel_final_dilate)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)