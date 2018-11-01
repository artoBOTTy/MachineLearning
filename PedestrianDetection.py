# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:29:02 2018

@author: Rakshith
"""

datadir =  "notebooks/data/chapter6"
dataset = "pedestrians128x64"
extractdir = 'notebooks/data/chapter6/pedestrians128x64/pedestrians128x64/'

import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from sklearn import model_selection as ms

#for i in range (5):
#    filename = "%s/pedestrians128x64/per0010%d.ppm" % (extractdir, i)
#    img = cv2.imread(filename)
#    plt.subplot(1, 5, i + 1)
#    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#    plt.axis('off')
    
win_size = (48, 96)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)


random.seed(42)
x_pos = []
for i in random.sample(range(900), 400):
    filename = "%s/per%05d.ppm" % (extractdir, i + 1)
    img = cv2.imread(filename)
    if img is None:
        print("Could not find image %s" % filename)
        continue
    x_pos.append(hog.compute(img, (64, 64)))
    
x_pos = np.array(x_pos, dtype=np.float32)
y_pos = np.ones(x_pos.shape[0], dtype=np.int32)
x_pos.shape, y_pos.shape


negdir = 'notebooks/data/chapter6/pedestrians_neg'
    
hroi = 128
wroi = 64
x_neg = []
for negfile in os.listdir(negdir):
    filename = '%s/%s' % (negdir, negfile)
    img = cv2.imread(filename)
    img = cv2.resize(img, (512, 512))
    for j in range(5):
        rand_y = random.randint(0, img.shape[0] - hroi)
        rand_x = random.randint(0, img.shape[1] - wroi)
        roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :]
    x_neg.append(hog.compute(roi, (64, 64)))
    
x_neg = np.array(x_neg, dtype=np.float32)
y_neg = -np.ones(x_neg.shape[0], dtype=np.int32)
x_neg.shape, y_neg.shape

x = np.concatenate((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))

win_size = (48, 96)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)


random.seed(42)
x_pos = []
for i in random.sample(range(900), 400):
    filename = "%s/per%05d.ppm" % (extractdir, i)
    img = cv2.imread(filename)
    if img is None:
        print("Could not find image %s" % filename)
        continue
    x_pos.append(hog.compute(img, (64, 64)))
    
x_pos = np.array(x_pos, dtype=np.float32)
y_pos = np.ones(x_pos.shape[0], dtype=np.int32)
x_pos.shape, y_pos.shape

negdir = 'notebooks/data/chapter6/pedestrians_neg'
    

hroi = 128
wroi = 64
x_neg = []
for negfile in os.listdir(negdir):
    filename = '%s/%s' % (negdir, negfile)
    img = cv2.imread(filename)
    img = cv2.resize(img, (512, 512))
    for j in range(5):
        rand_y = random.randint(0, img.shape[0] - hroi)
        rand_x = random.randint(0, img.shape[1] - wroi)
        roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :]
    x_neg.append(hog.compute(roi, (64, 64)))
    
x_neg = np.array(x_neg, dtype=np.float32)
y_neg = -np.ones(x_neg.shape[0], dtype=np.int32)
x_neg.shape, y_neg.shape

x = np.concatenate((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.2, random_state=42)

def train_svm(x_train, y_train):
    svm = cv2.ml.SVM_create()
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm

def score_svm(svm, x, y):
    from sklearn import metrics
    _, y_pred = svm.predict(x)
    return metrics.accuracy_score(y, y_pred)

svm = train_svm(x_train, y_train)
score_svm(svm, x_train, y_train)
score_svm(svm, x_test, y_test)

img_test = cv2.imread('notebooks/data/chapter6/people_test_005.jpg')
plt.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))

stride = 16
found = []
for ystart in np.arange(0, img_test.shape[0], stride):
    for xstart in np.arange(0, img_test.shape[1], stride):
        if ystart + hroi > img_test.shape[0]:
            continue
        if xstart + wroi > img_test.shape[1]:
            continue
        roi = img_test[ystart:ystart + hroi, xstart:xstart + wroi, :]
        feat = np.array([hog.compute(roi, (64, 64))])
        _, ypred = svm.predict(feat)
        if np.allclose(ypred, 1):
            found.append((ystart, xstart, hroi, wroi))
            
rho, _, _ = svm.getDecisionFunction(0)

sv = svm.getSupportVectors()

print(x_neg.shape, y_neg.shape)

hogdef = cv2.HOGDescriptor()
pdetect = cv2.HOGDescriptor_getDefaultPeopleDetector()

hogdef.setSVMDetector(pdetect)

found, _ = hogdef.detectMultiScale(img_test)

from matplotlib import patches
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
for f in found:
    ax.add_patch(patches.Rectangle((f[0], f[1]), f[2], f[3], color='y', linewidth=3, fill=False))