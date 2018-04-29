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

XML_DIR = 'C:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations'
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
        'horse': 20}
N_LEN = 1
C_LEN = 20
IMG_SIZE = (224, 224)

num_images = len(os.listdir())
training_y = np.empty([num_images, 7, 7, 25])
rects = []
for count, fname in enumerate(os.listdir()):
    target = np.zeros([7,7,25])
    tree = etree.parse(fname)
    root = tree.getroot()
    size = root.find('size')
    # Get height and width so we can scale appropriately
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    #obj = root.find('object')
    divfac = len(root.findall('object'))
    for obj in root.findall('object'):
        name = obj.find('name').text
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        xmax = float(bb.find('xmax').text)
        ymin = float(bb.find('ymin').text)
        ymax = float(bb.find('ymax').text)
        

        # high confidence in all boxes object touches - smoothed for specific category
        for i in range(int(xmin / (w / 7)),int(xmax / (w / 7))+1):
            for j in range(int(ymin / (h / 7)),int(ymax / (h / 7)) +1):
                try:
                    target[j,i,0] = 1 #general conf
                    target[j,i,4 + labels[name]] = 1 / max(divfac,3) # category conf

                    # scale coords to new img
                    target[j,i,1] = xmin * 224 / w# x
                    target[j,i,2] = (xmax - xmin) * 224 / w# width
                    target[j,i,3] = ymin * 224 / h# y
                    target[j,i,4] = (ymax - ymin) * 224 / h #height
                    rects.append([target[j,i,1], target[j,i,3], target[j,i,2], target[j,i,4]])
                except:
                    pass
        # insert positive label for confidence at center
        centerx = int(0.5 * (xmin + xmax)/ (w / 7) ) 
        centery = int(0.5 * (ymin + ymax)/ (h / 7) ) 
        target[centery, centerx, 4 + labels[name]] = 1
        target[centery,centerx,0] = 1 #general conf
        

        
        # insert it into the training array
        #training_y[i,:,:,:] = target
        # Save as individual numpy array for data generator
    np.save('D:/VOC_individual_np/labels/{}.npy'.format(count), target)
        

## TOM UPDATE
for count, fname in enumerate(os.listdir()):
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
    np.save('C:/VOC_individual_np/labels/{}.npy'.format(fname[:-4]), target)
    if count % 1000 == 0:
        print(count)
        
# Gathering images
IMG_DIR = 'D:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'
num_images = len(os.listdir())
#train_x = np.empty([int(num_images/4), 224, 224, 3])
#os.chdir(IMG_DIR)
#avg_colors = (124.59085262283071, 124.91715538568295, 124.90344722141644) # precomputed on the training set in bgr order
#count = 0
#for i, img in enumerate(os.listdir()):
#    img = cv2.imread(img)
#    img = cv2.resize(img, (224,224))
#    img = img - avg_colors
#    np.save('C:/VOC_individual_np/images/{}.npy'.format(i), img)
#
#    
#    
    
    
    # DISPLAY PURPOSES
    
IMG_DIR = 'D:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages'
os.chdir(IMG_DIR)
img = cv2.imread(os.listdir()[1])
img = cv2.resize(img, (224,224))
alpha = 0.4
heatmap = target[:,:,0]
heatmap = cv2.resize(heatmap, (224,224),0,0,0,cv2.INTER_NEAREST)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
output = img.copy()
cv2.addWeighted(heatmap, alpha, output, 1 - alpha, 0, output)
for i in range(0,224,int(224/7)):
    cv2.line(output,(0, i), (224, i), (0,255,0),1 )
    cv2.line(output,(i,0), (i,224), (0,255,0),1 )
for rect in rects:
    x, y, w, h = [int(x) for x in rect]
    cv2.rectangle(output, (x,y),(x+w,y+h), (0,0,0), 1)
cv2.imshow('super', cv2.resize(np.uint8(output), (500,500))); cv2.waitKey()
    
    
    
    
    
    
    
    
    