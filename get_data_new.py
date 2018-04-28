# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:38:21 2018

@author: ander
"""

import xml.etree.ElementTree as etree
import os
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm

BASE_DIR = "M:/Documents/Courses/CSE586/finalProject/CV2_final_project"
N_LEN = 1
C_LEN = 20
IMG_SIZE = (224, 224)

XML_DIR = os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/Annotations")
os.chdir(XML_DIR)
labels = {
    'bird': 1,
    'bicycle': 2,
    'chair': 3,
    'pottedplant': 4,
    'cat': 5,
    'car': 6,
    'cow': 7,
    'person': 8,
    'tvmonitor': 9,
    'sofa': 10,
    'motorbike': 11,
    'sheep': 12,
    'bus': 13,
    'train': 14,
    'boat': 15,
    'dog': 16,
    'bottle': 17,
    'diningtable': 18,
    'aeroplane': 19,
    'horse': 20
}

num_images = len(os.listdir())
for count, fname in enumerate(tqdm(iterable=os.listdir(), total=num_images)):
    
    # get some info on the current file
    tree = etree.parse(fname)
    root = tree.getroot()

    # initialize the target to be predicted
    target = np.zeros((1, N_LEN + C_LEN))

    # find all objects in the image
    object_count = defaultdict(lambda: 0)
    num_objects = 0
    for obj in root.findall('object'):
        name = obj.find('name').text
        object_count[name] += 1
        num_objects += 1

    # find most common object
    max_count = -1
    max_obj = ""
    for key in labels.keys():
        obj_count = object_count[key]
        if obj_count > 0:
            max_count = obj_count
            max_obj = key
        
            # add the obj to the target for prediction
            # note that this will give multiple predicted targets
            # we take the max and testing time
            target[0, labels[key] + N_LEN - 1] += 1

    target[0, 1:] = (np.exp(target[0, 1:])-1) / np.sum(np.exp(target[0, 1:])-1, axis=0)
    target[0, 0] = num_objects
    np.save(os.path.join(BASE_DIR, "data/preprocessed_2/labels/{}.npy".format(count)), target)

# Gathering images
# IMG_DIR = os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages")
# num_images = len(os.listdir())
# #train_x = np.empty([int(num_images/4), IMG_SIZE[0], IMG_SIZE[1], 3])
# os.chdir(IMG_DIR)
# avg_colors = (124.59085262283071, 124.91715538568295, 124.90344722141644) # precomputed on the training set in bgr order
# count = 0
# for i, img in enumerate(os.listdir()):
#     img = cv2.imread(img)
#     img = cv2.resize(img, IMG_SIZE)
#     img = img - avg_colors
#     np.save(os.path.join(BASE_DIR, "data/preprocessed/images/{}.npy".format(i), img))


# # DISPLAY PURPOSES
# target = np.load(os.path.join(BASE_DIR, "data/preprocessed_2/labels/1.npy"))
# IMG_DIR = os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages")
# os.chdir(IMG_DIR)
# img = cv2.imread(os.listdir()[1])
# img = cv2.resize(img, IMG_SIZE)
# alpha = 0.4
# heatmap = target[:,:,0]
# heatmap = cv2.resize(heatmap, IMG_SIZE,0,0,0,cv2.INTER_NEAREST)
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# output = img.copy()
# cv2.addWeighted(heatmap, alpha, output, 1 - alpha, 0, output)
# for i in range(0,IMG_SIZE[0],int(IMG_SIZE[0]/S)):
#     cv2.line(output,(0, i), (IMG_SIZE[0], i), (0,255,0),1 )
#     cv2.line(output,(i,0), (i,IMG_SIZE[1]), (0,255,0),1 )
# for rect in rects:
#     x, y, w, h = [int(x) for x in rect]
#     cv2.rectangle(output, (x,y),(x+w,y+h), (0,0,0), 1)
# cv2.imshow('super', cv2.resize(np.uint8(output), (500,500))); cv2.waitKey()