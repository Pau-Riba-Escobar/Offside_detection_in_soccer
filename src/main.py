import numpy as np
import os
import cv2

"""
def getVP(im):
    edges = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY),500, 460)
    lines = cv2.HoughLines(edges,1, np.pi/180,240)
    for l in lines:
        r,t = l[0]
        s,c = [np.sin(t),np.cos(t)]
        # Central point of the line
        xb,yb = [c, s]*r
        # getting 2 points to create the line
        x1,y1 = [int(xb + 1000*(-s)),int(yb+1000*c)]
        x2,y2 = [int(xb - 1000*(-s)),int(yb-1000*c)]
        cv2.line(im,(x1,y1),(x2,y2),(0,255,0),1)
"""
def drawLines(im,edges):
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    # taking the most important lines
    for l in lines[1:5]:
        r,t = l[0]
        s,c = [np.sin(t),np.cos(t)]
        # Central point of the line
        xb,yb = [c*r, s*r]
        # getting 2 points to create the line
        x1,y1 = [int(xb + 10000*(-s)),int(yb + 10000*c)]
        x2,y2 = [int(xb - 10000*(-s)),int(yb - 10000*c)]
        cv2.line(im,(x1,y1),(x2,y2),(255,0,0),1)
    # image with lines
    return im




# set de imatges
imset = ["Dataset/"+file for file in os.listdir("Dataset/")]

# imatge d'exemple
im = cv2.imread(imset[0])
new_size = (int(im.shape[0]/2),int(im.shape[1]/2))
# imatge més petita per a un millor processament
im = cv2.resize(im,new_size)
# Imatge amb HSV per a obtenir millor els edges, ja que les línies blanques no es veuen gaire bé en algunes imatges
hsv_im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
edges = cv2.Canny(hsv_im,150,250,apertureSize=3)
im_with_lines = drawLines(im, edges)
cv2.imshow("image with lines",im_with_lines)
cv2.waitKey(0)





