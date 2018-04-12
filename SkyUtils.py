# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:37:05 2018

@author: ander
"""
import cv2
import numpy as np
from keras import backend as K
import keras

def drawRectsFromPred(predictions, img):
    for i in range(7):
        for j in range(7):
            print(predictions[i,j,0])
            #consider areas where we get a >0.3 confidence that there is an object
            if predictions[i,j,0] > 0.3:
                rect = [predictions[j,i,1], predictions[j,i,3], predictions[j,i,2], predictions[j,i,4]]
                x, y, w, h = [int(x) for x in rect]
                print(x,y,w,h)
                cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 1)
    return img


def customloss(gt, pred):
    """ Custom loss function to apply categorical cross entropy to the classification
    results and L2 loss to confidence and box coordinates"""
    #assert gt.shape == pred.shape
    
    loss = K.variable(0.,dtype='float32')
    print(gt.shape)
    print(pred.shape)
    try: 
        batch_size = int(gt.shape[0])
    except:
        batch_size = 32
        
    for batch in range(batch_size):
        for i in range(7):
            for j in range(7):
                # Catgeorical crossentropy for classwise confidence
                loss = K.add(loss , K.sum(K.categorical_crossentropy(gt[batch,i,j,5:], pred[batch,i,j,5:])))
                # MSE for confidence and bounding box predictions
                loss = K.add(loss , K.mean(K.square(gt[i,j,:5] - pred[i,j,:5])))
            
    return loss / (batch_size * 25 * 25)