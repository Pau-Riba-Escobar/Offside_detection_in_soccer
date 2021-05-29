import numpy as np
import cv2
from matplotlib import pyplot as plt

lower_green = np.array([36,0,0])
upper_green = np.array([86, 255, 255])


player_k = np.ones((5,5),np.uint8)
audience_k = np.ones((30,30),np.uint8)
closing_k = np.ones((40,40),np.uint8) 
erode_k = np.ones((15,1),np.uint8)
dilate_k = np.ones((10,10),np.uint8)

# function that extracts the players, the gk and the referee using a preset range of green as a mask for field segmentation
def extract_playersV1(im):
    # hsv
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, lower_green, upper_green)
    cv2.imshow('green_mask',grass_mask)
    cv2.waitKey(0)
    # FER CLOSING PER TAPAR ELS JUGADORS
    # RESTAR ELS JUGADORS DE LA IMATGE INICIAL
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

def getHueThresh(hsv_im):
    # calculating a histogram of hue component( color)
    hist = cv2.calcHist([hsv_im], [0],mask = None, histSize=[360], ranges=[0,360]) # histSzie= number of bins
    # taking only the interesting part of it
    max_val = np.argmax(hist) 
    init_l = max_val
    init_r = max_val
    diff = np.inf
    dev = 0
    while diff > 0:
        diff = (hist[init_l] - hist[init_l - 1]) + (hist[init_r] - hist[init_r + 1]) 
        init_l -= 1
        init_r += 1
        dev += 1
    # plt.plot(hist[max_val - dev:max_val + dev], color="g")
    plt.plot(hist, color="g")
    plt.show()
    return np.array([max_val-dev,0,0]),np.array([max_val+dev, 255, 255])

def extract_playersv2(im):
    # im to hsv to extract hue info
    hsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    # getting a threshold of the hue of the image and taking the largest component as grass
    lower_bound, upper_bound  = getHueThresh(hsv)
    grass_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # dilating to remove lines noise
    gmask_dilated = cv2.morphologyEx(grass_mask, cv2.MORPH_DILATE, np.ones((3,3)))
    # gmask_dilated = cv2.morphologyEx(gmask_dilated, cv2.MORPH_ERODE, np.ones((15,30)))
    cv2.imshow("grass mask applied", gmask_dilated)
    # cv2.waitKey(0)
    audience_open = cv2.morphologyEx(grass_mask,cv2.MORPH_CLOSE,np.ones((15,15)))
    audience = cv2.morphologyEx(audience_open,cv2.MORPH_OPEN,np.ones((70,70)))
    pitch = 255 - audience
    cv2.imshow("audience out field in", pitch)
    # bitwise and to put together both masks
    m = cv2.bitwise_and(audience, gmask_dilated)
    m = cv2.bitwise_or(m,pitch)
    cv2.imshow("clean mask", m)
    cv2.waitKey(0)
    return m 
    # REGION PROPS
    # nlabels,labels,stats,centroids = cv2.connectedComponentsWithStats(255-tmp)
    # bg_mask = (labels == 0).astype("uint8")*255
    # tmp2 = cv2.bitwise_and(tmp,bg_mask)
    # removing the audience with an opening
    

def show_contours(im, thresh): 
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("contoured",im)
    cv2.waitKey(0)
        