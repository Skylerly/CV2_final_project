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
    S = 7
    for i in range(S):
        for j in range(S):
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
    # gt stands for ground truth
    # pred is the predicted values

    print("ground truth shape: {}".format(gt))
    print("pred shape: {}".format(pred))

    S = 7
    B = 1
    C_LEN = 20
    
    loss = K.variable(0.,dtype='float32')
    print(gt.shape)
    print(pred.shape)
    try: 
        batch_size = int(gt.shape[0])
    except:
        batch_size = 4

    # set some parameters
    lam_coord = 5.0
    lab_noobj = 0.5

    # loss from bouding box locations
    loss_bbloc = 0
    loss += loss_bbloc

    # loss from boudning box sizes
    loss_bbsize = 0
    loss += loss_bbsize

    # loss from bounding box predictor responsible
    # for prediction
    loss_bbpred = 0
    loss += loss_bbpred

    # loss from boudning box predictor that is
    # not responsible for prediction
    loss_bbnopred = 0
    loss += loss_bbnopred

    # loss from probabilities for cells where the given
    # object appears
    loss_prob = 0
    loss += loss_prob

    # something like:
    #     conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    # prediction = prediction*conf_mask
        
    # for batch in range(batch_size):
    #     for i in range(7):
    #         for j in range(7):
    #             # Catgeorical crossentropy for classwise confidence
    #             loss = K.add(loss , K.sum(K.categorical_crossentropy(gt[batch,i,j,5:], pred[batch,i,j,5:])))
    #             # MSE for confidence and bounding box predictions
    #             loss = K.add(loss , K.mean(K.square(gt[i,j,:5] - pred[i,j,:5])))
            
    #return loss / (batch_size * (C_LEN + 5*B) * (C_LEN + 5*B))
    return K.square(-pred - gt)