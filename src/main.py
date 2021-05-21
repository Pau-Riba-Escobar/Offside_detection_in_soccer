import numpy as np
import os
import cv2
from numpy.lib import utils
from geometric_utils import *

"""
def intersection(lines, points):
    for l in lines:
       x = (l[0][0][0] - l[0][1][0], points[0][0]-points[1][0])
       y =(l[0][0][1] - l[0][1][1], points[0][1]-points[1][1])
       # Per resoldre el sistema d'equacions , si no té solució no hi ha intersecció entre les rectes. El determinant ha de ser > 0
       # per trobar solució
       if (x[0] * y[1] - x[1] * y[0]) != 0:
           return True,l
    return False, None

# NOMÉS TINDREM EN COMPTE LÍNIES VERTICALS
def get_lines(im, lines, orientation):
    if orientation == "right":
        max_angle =  150
        min_angle = 105
    else:
        max_angle =  70
        min_angle = 30
    # Creem una variable per guardar les línies que seràn bones
    good_lines = []

    # CAS DE LA PRIMERA LÍNEA
    # guardem comptador per saber per on continuar

    for last_line,l in enumerate(lines):
        # guardem ro i theta de cada una de les linies
        r,t = l[0]
        angle = t*180/np.pi
        # si theta se encuentra dentro del rango indicado
        if  angle > min_angle and angle < max_angle:
            s,c = [np.sin(t),np.cos(t)]
            # Central point of the line
            xb,yb = [c*r, s*r]
            # Necessitem dos punts extrems per a construir la línia
            p1 = [int(xb + 10000*(-s)),int(yb + 10000*c)]
            p2 = [int(xb - 10000*(-s)),int(yb - 10000*c)]
            good_lines.append([[p1,p2],[r,t]])
            cv2.line(im,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,0),1)
            break
    
    rthd = 50
    # CAS DE LA SEGONA LÍNEA
    
    for l in lines[last_line+1:]:
        
        # guardem ro i theta de cada una de les linies
        r,t = l[0]
        angle = t*180/np.pi
        # si theta se encuentra dentro del rango indicado
        if  angle > min_angle and angle < max_angle:
            s,c = [np.sin(t),np.cos(t)]
            # Central point of the line
            xb,yb = [c*r, s*r]
            # Necessitem dos punts extrems per a construir la línia
            p1 = [int(xb + 10000*(-s)),int(yb + 10000*c)] # point 1=(x1,y1)
            p2 = [int(xb - 10000*(-s)),int(yb - 10000*c)] # point 2=(x2,y2)
            intersected, intersected_line = intersection(good_lines,[p1,p2])
            # Si les línies han intersectat i estàn suficientment separades(no volem linies quasi coincidents)
            if intersected and abs(intersected_line[1][0] - r) > 50:
                good_lines.append([[p1,p2],[r,t]])
                cv2.line(im,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,0),1)
                break
    # image with lines
    cv2.imshow("lines of the image", im)
    cv2.waitKey(0)
    return good_lines

orientation = "left"
# set de imatges
imset = ["../Dataset/"+file for file in os.listdir("../Dataset/")]

# imatge d'exemple
im = cv2.imread(imset[11])
# old_shape = (cv2.resize(im,(int(im.shape[0]/2), int(im.shape[1]/2)))).shape
new_size = (int(im.shape[0]/2), int(im.shape[1]/2))
# imatge més petita per a un millor processament
im = cv2.resize(im, new_size)
# Imatge amb HSV per a obtenir millor els edges, ja que les línies blanques no es veuen gaire bé en algunes imatges
hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
# edges = cv2.Canny(hsv_im, 150, 250, apertureSize=3)
edges = cv2.Canny(hsv_im, 150, 250, apertureSize=5)
tmp_lines = cv2.HoughLines(edges,1,np.pi/180,200)
lines = get_lines(im, tmp_lines, orientation)
"""
# image set
imset = ["../Dataset/train/"+file for file in os.listdir("../Dataset/train/")]

test_im = cv2.imread(imset[2])
# resizing the image for testing to half of its size
test_im = cv2.resize(test_im,(int(test_im.shape[0]/2), int(test_im.shape[1]/2)))
# T,thresholded_im = cv2.threshold(cv2.cvtColor(test_im,cv2.COLOR_BGR2GRAY),200, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh_im",thresholded_im)
# cv2.waitKey(0)
# Now we have to apply hough transform to obtain the lines of the image
lines = get_lines(test_im)
vp = get_vanishing_point(lines)
print("vanishing point", vp)
"""

detected = 0
for im in imset:
    test_im = cv2.imread(im)
    # resizing the image for testing to half of its size
    test_im = cv2.resize(test_im,(int(test_im.shape[0]/2), int(test_im.shape[1]/2)))
    # Now we have to apply hough transform to obtain the lines of the image
    lines = get_lines(test_im)
    detected = detected+1 if len(lines) > 1 else detected

print("accuracy: ", detected/len(imset))
"""

# to watch what lines we have found let's draw them in our test image and show them
im = draw_lines(test_im,lines)
cv2.imshow("lined image", im)
cv2.waitKey(0)



