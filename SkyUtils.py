# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:37:05 2018

@author: ander
"""
import cv2
import numpy as np
from keras import backend as K
import keras
from keras import losses, metrics
import os
import tensorflow as tf

VOC_labels = {
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
VOC_labels_flipped = {}
for key,val in VOC_labels.items():
    VOC_labels_flipped[val] = key

def makePredictions(model, image_dir, label_dir, indices):
    # constants
    N_LEN = 1
    C_LEN = 20
    IMG_SIZE = (224, 224, 3)
    # precomputed on the training set in bgr order
    avg_colors = (124.59085262283071, 124.91715538568295, 124.90344722141644)

    # get list of all images and labels
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    # loop over samples to predict
    for idx in indices:
        # load data
        img = cv2.imread(os.path.join(image_dir, images[idx]))
        img = cv2.resize(img, IMG_SIZE[0:2])
        img = img - avg_colors
        target = np.load(os.path.join(label_dir, labels[idx]))

        # make prediction
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        
        # output results
        print("---------------------- Image: {} ({}) ----------------------".format(idx, images[idx]))
        print("Target: {}".format(target))
        print("Prediction: {}".format(pred))
        # pred = np.expand_dims(pred, axis=0)
        # target = np.expand_dims(target, axis=0)
        # print("MSE loss: {}".format(np.mean((target[:, 0] - pred[:, 0])**2)))
        # print("Categorical loss: {}".format(losses.categorical_crossentropy(target[:, 1:], pred[:, 1:])))
        # print("Total Loss: {}".format(custom_loss_2(target, pred)))
        # print("Custom MSE accuracy: {}".format(custom_accuracy_num(target, pred)))
        # print("Custom cat accuracy: {}".format(custom_accuracy_cat(target, pred)))

def makePredictions2(model, image_dir, label_dir, indices):
    # constants
    N_LEN = 1
    C_LEN = 20
    IMG_SIZE = (224, 224, 3)
    # precomputed on the training set in bgr order
    avg_colors = (124.59085262283071, 124.91715538568295, 124.90344722141644)

    # get list of all images and labels
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)

    # loop over samples to predict
    for idx in indices:
        # load data
        img = cv2.imread(os.path.join(image_dir, images[idx]))
        img = cv2.resize(img, IMG_SIZE[0:2])
        img = img - avg_colors
        target = np.load(os.path.join(label_dir, labels[idx]))
        target = target[0]
#        print(idx)
#        print(os.path.join(label_dir, labels[idx]))
#        print(target)
#        print(os.path.join(image_dir, images[idx]))
        # make prediction
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        
        #clean up pred so its easier to look at
        pred = pred[0]
#        print(pred)
        for i,val in enumerate(pred):
            pred[i] = round(val,2)
       
        # output results
        print("---------------------- Image: {} ({}) ----------------------".format(idx, images[idx]))
        print("           | Target   | Predicted")
        print("Num Objects|   {}    |   {}     ".format(target[0],round(pred[0])))
        for i,val in enumerate(pred[1:]):
            # if target[i+1] > 0:
            print(VOC_labels_flipped[i+1] + " " * (11 - len(VOC_labels_flipped[i+1])) + "|   {:.2f}   |   {:.2f}     ".format(target[i+1],round(pred[i+1],2)) )



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

def custom_loss_2(gt, pred):

    # loss from total number of objects
    num_objects_loss_scalar = 2.0
    num_objects_loss = losses.mean_squared_error(gt[:, 0], pred[:, 0])

    # figure out how to predict loss for the categorical part
    # do I need to convert it to binary and one class
    # to use categorical cross entropy?
    categorical_loss_scalar = 3.0
    categorical_loss = losses.categorical_crossentropy(gt[:, 1:], pred[:, 1:])

    total_loss = num_objects_loss_scalar * num_objects_loss + \
        categorical_loss_scalar  * categorical_loss
    # total_loss = num_objects_loss

    return total_loss

def custom_accuracy_cat(gt, pred):
    target = gt[:, 1:] / K.sum(K.square(gt[:, 1:]), axis=-1)
    predicted = pred[:, 1:] / K.sum(K.square(pred[:, 1:]), axis=-1)
    multiplied = keras.layers.multiply([target, predicted])
    return K.sum(K.flatten(multiplied))

def custom_accuracy_top_5(gt, pred):
    return metrics.top_k_categorical_accuracy(gt[:, 1:], pred[:, 1:], k=5)

def custom_accuracy_top_1(gt, pred):
    return metrics.top_k_categorical_accuracy(gt[:, 1:], pred[:, 1:], k=1)

def custom_accuracy_top_10(gt, pred):
    return metrics.top_k_categorical_accuracy(gt[:, 1:], pred[:, 1:], k=10)

def custom_accuracy_num(gt, pred):
    return metrics.mean_squared_error(gt[:, 0], pred[:, 0])

def customloss(gt, pred):
    """ Custom loss function to apply categorical cross entropy to the classification
    results and L2 loss to confidence and box coordinates"""
    # gt stands for ground truth
    # pred is the predicted values

    print("ground truth shape: {}".format(gt))
    print("pred shape: {}".format(pred))

    S = 7
    B = 1
    C_LEN = 20
    CONFIDENCE_THRESH = 0.5

    # reshape the input to make it easier to calculate loss
    # MAKE SURE THAT THESE ARE ACCURATE
    y_true = K.reshape(gt, shape=[-1, B*5+C_LEN])
    y_pred = K.reshape(pred, shape=[-1, B*5+C_LEN])
    print(K.eval(y_true))
    print(K.eval(y_pred))

    #t = K.greater(truth_confid_tf, 0.5) 
    confidence_mask = K.tf.where(y_true > CONFIDENCE_THRESH)
    
    #loss = K.variable(0.,dtype='float32')
    # print("Ground Truth shape: {}".format(gt.shape))
    # print("Pred shape: {}".format(pred.shape))
    try: 
        batch_size = int(gt.shape[0])
    except:
        batch_size = 4

    # prediction output: (confidence, x, w, y, h, probs)

    # set some parameters
    lam_coord = 5.0
    lab_noobj = 0.5

    # loss from bouding box locations
    # FIND THE ACTUAL MASK
    loss_bbloc_mask = K.zeros(shape=(batch_size, S, S))
    x_pred, y_pred = (pred[:, :, :, 1], pred[:, :, :, 3])
    x_gt, y_gt = (gt[:, :, :, 1], gt[:, :, :, 3])
    loss_bbloc = loss_bbloc_mask * (K.square(x_pred-x_gt) + K.square(y_pred-y_gt))
    loss.assign_add(K.sum(K.reshape(loss_bbloc, shape=(-1, 1))))

    # loss from boudning box sizes
    # FIND THE ACTUAL MASK
    loss_bbsize_mask = K.zeros(shape=(batch_size, S, S))
    w_pred, h_pred = (pred[:, :, :, 2], pred[:, :, :, 4])
    w_gt, h_gt = (gt[:, :, :, 2], gt[:, :, :, 4])
    w_pred = K.sqrt(w_pred)
    w_gt = K.sqrt(w_gt)
    h_pred = K.sqrt(h_pred)
    h_gt = K.sqrt(h_gt)
    loss_bbsize = loss_bbsize_mask * (K.square(w_pred-w_gt) + K.square(h_pred-h_gt))
    loss.assign_add(K.sum(K.reshape(loss_bbsize, shape=(-1,1))))

    # loss from bounding box predictor responsible
    # for prediction
    # FIND THE ACTUAL MASK
    loss_bbpred_mask = K.zeros(shape=(batch_size, S, S))
    loss_bbpred = loss_bbpred_mask
    loss.assign_add(K.sum(K.reshape(loss_bbpred, shape=(-1,1))))

    # loss from boudning box predictor that is
    # not responsible for prediction
    # FIND THE ACTUAL MASK
    loss_bbnopred_mask = K.ones(shape=(batch_size, S, S)) - loss_bbpred_mask
    loss_bbnopred = loss_bbnopred_mask 
    loss.assign_add(K.sum(K.reshape(loss_bbnopred, shape=(-1, 1))))

    # loss from probabilities for cells where the given
    # object appears
    # FIND THE ACTUAL MASK
    # NEED TO ADD A SIGMOID
    loss_prob_mask = K.ones(shape=(batch_size, S, S))
    probs_pred = pred[:, :, :, 5:]
    probs_gt = gt[:, :, :, 5:]
    loss_prob = loss_prob_mask * losses.categorical_crossentropy(probs_gt, probs_pred)
    loss.assign_add(K.sum(K.reshape(loss_prob, shape=(-1,1))))

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
    # loss = K.square(pred - gt)
    # print("Loss: {}".format(loss))
    return loss