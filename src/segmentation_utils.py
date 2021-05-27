import numpy as np
import cv2

#converting into hsv image
#green range
lower_green = np.array([36,0,0])
upper_green = np.array([86, 255, 255])


player_k = np.ones((5,5),np.uint8)
audience_k = np.ones((30,30),np.uint8)
closing_k = np.ones((40,40),np.uint8) 
erode_k = np.ones((15,1),np.uint8)
dilate_k = np.ones((10,10),np.uint8)

def extract_players(im):
    # hsv
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('green_mask',grass_mask)
    cv2.waitKey(0)
    mask = 255 - grass_mask
    thresh_players = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, player_k)
    cv2.imshow('players',thresh_players)
    cv2.waitKey(0)
    thresh_aud = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, audience_k)
    cv2.imshow('aud',thresh_aud)
    cv2.waitKey(0)
    thresh_aud = cv2.morphologyEx(thresh_aud, cv2.MORPH_OPEN, closing_k)
    cv2.imshow('aud2',thresh_aud)
    cv2.waitKey(0)
    thresh_aud = 255 - thresh_aud
    comb_thresh = cv2.bitwise_and( thresh_aud , thresh_players)
    thresh_no_noise = cv2.morphologyEx(comb_thresh,cv2.MORPH_ERODE,erode_k)
    thresh = cv2.morphologyEx(thresh_no_noise,cv2.MORPH_DILATE,dilate_k)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
    return thresh

def show_contours(im, thresh): 
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("contoured",im)
    cv2.waitKey(0)
        