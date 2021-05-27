import numpy as np
import os
import cv2
from numpy.lib import utils
from geometric_utils import *

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

# to watch what lines we have found let's draw them in our test image and show them
im = draw_lines(test_im,lines)
cv2.imshow("lined image", im)
cv2.waitKey(0)



