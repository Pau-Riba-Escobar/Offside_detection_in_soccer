import numpy as np
import os
import cv2
from numpy.lib import utils
from geometric_utils import *
from segmentation_utils import *

# image set
imset = ["DataSet/Train/"+file for file in os.listdir("DataSet/Train/")]

test_im = cv2.imread(imset[3])
# resizing the image for testing to half of its size
test_im = cv2.resize(test_im,(int(test_im.shape[0]/2), int(test_im.shape[1]/2)))
# Now we have to apply hough transform to obtain the lines of the image
# 1. VANISHING POINT
lines = get_lines(test_im)
vp = get_vanishing_point(lines)

# to watch what lines we have found let's draw them in our test image and show them
# im = draw_lines(test_im,lines) UNCOMMENT TO SEE THE LINES IN THE IMAGE
extract_players(test_im)



