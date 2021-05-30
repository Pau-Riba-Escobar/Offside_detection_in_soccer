import numpy as np
import os
import cv2
from numpy.lib import utils
from numpy.random import triangular
from geometric_utils import *
from segmentation_utils import *
from team_classification import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

# image set
imset = ["DataSet/Train/"+file for file in os.listdir("DataSet/Train/")]

# test_im = cv2.imread(imset[0])
# # resizing the image for testing to half of its size
# test_im = cv2.resize(test_im,(int(test_im.shape[0]/4), int(test_im.shape[1]/4)))
# # Now we have to apply hough transform to obtain the lines of the image

# # 1. VANISHING POINT

# lines = get_lines(test_im)
# vp = get_vanishing_point(lines)

# thr = extract_playersV1(test_im)
# contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# show_contours(test_im,contours)
# to watch what lines we have found let's draw them in our test image and show them
# im = draw_lines(test_im,lines) UNCOMMENT TO SEE THE LINES IN THE IMAGE

# 2. GENERATING TRAIN SET

train_data = []
for im in imset:
    test_im = cv2.imread(im)
    # resizing the image for testing to half of its size
    test_im = cv2.resize(test_im,(int(test_im.shape[0]/4), int(test_im.shape[1]/4)))
    # lines = get_lines(test_im)
    # draw_lines(test_im.copy(),lines)
    thr = extract_playersV1(test_im)
    contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # show_contours(test_im.copy(), contours)
    # print(im)
    desc = getMeanDescriptors(test_im.copy(), contours)
    train_data.append(desc)

# train_data = np.array(train_data)
train_data =  np.concatenate(train_data, axis = 0)
# train_data = train_data.reshape(train_data.shape[0], train_data.shape[1])
train_labels = np.genfromtxt("train_labels.txt", delimiter=",").astype("uint8")
train_data = np.array(train_data).reshape((len(train_labels), -1))

classifier = OneVsRestClassifier(estimator=SVC())# KMeans(n_clusters=3,random_state=5)
classifier.fit(train_data, train_labels)

# TEST DATA
imset = ["DataSet/Test/"+file for file in os.listdir("DataSet/Test/")]
test_data = []
for im in imset:
    test_im = cv2.imread(im)
    # resizing the image for testing to half of its size
    test_im = cv2.resize(test_im,(int(test_im.shape[0]/4), int(test_im.shape[1]/4)))
    thr = extract_playersV1(test_im)
    contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # show_contours(test_im.copy(), contours)
    desc = getMeanDescriptors(test_im.copy(), contours)
    test_data.append(desc)

test_data =  np.concatenate(test_data, axis = 0)
test_labels = np.genfromtxt("test_labels.txt", delimiter=",").astype("uint8")
test_data = np.array(test_data).reshape((len(test_labels), -1))
predicted_labels = classifier.predict(test_data)
# show_contours(test_im,contours,predicted_labels,n_colors=3)
acc = np.sum(predicted_labels == test_labels)/len(test_labels)
print("accuracy: ",acc)

# PRUEBAS LINEA FUERA DE JUEGO

imset = ["DataSet/Train/"+file for file in os.listdir("DataSet/Train/")]
test_im = cv2.imread(imset[0])
# resizing the image for testing to half of its size
test_im = cv2.resize(test_im,(int(test_im.shape[0]/4), int(test_im.shape[1]/4)))
lines = get_lines(test_im)
vp = get_vanishing_point(lines)
# cv2.circle(test_im,(int(vp[1]),int(vp[0])),radius=4,color=2,thickness=1)
# cv2.imshow("im",test_im)
# cv2.waitKey(0)



