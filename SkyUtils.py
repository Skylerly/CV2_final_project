# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 23:37:05 2018

@author: ander
"""
import cv2
import numpy as np
from keras import backend as K
import keras
import tensorflow as tf
import random
from keras import losses, metrics
import os
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
        
        #clean up pred so its easier to look at
        pred = pred[0]
        for i,val in enumerate(pred):
            pred[i] = round(val,2)
        # output results
        print("---------------------- Image: {} ({}) ----------------------".format(idx, images[idx]))
        print("Target: {}".format(target))
        print("Prediction: {}".format(pred))

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
            if target[i+1] > 0:
                print(VOC_labels_flipped[i+1] + " " * (11 - len(VOC_labels_flipped[i+1])) + "|   {:.2f}   |   {:.2f}     ".format(target[i+1],round(pred[i+1],2)) )



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

def overlayPredsByColor(predictions,img):
    label_colors = {}
    preds = predictions[:,:,5:]
    colormap = np.zeros((7,7,3),dtype='uint8')
    for i in range(20):
        label_colors[i] = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        
    for i in range(7):
        for j in range(7):
            colormap[i,j,:] = label_colors[np.argmax(preds[i,j,:])]
            
    alpha = 0.4
    colormap = cv2.resize(colormap, (224,224),0,0,0,cv2.INTER_NEAREST)
    output = img.copy()
    output = cv2.addWeighted(colormap, alpha, output, 1 - alpha, 0, output)
    return output
    


def custom_loss_2(gt, pred):

    # loss from total number of objects
    num_objects_loss_scalar = 5.0
    num_objects_loss = losses.mean_squared_error(gt[:, 0], pred[:, 0])

    # figure out how to predict loss for the categorical part
    # do I need to convert it to binary and one class
    # to use categorical cross entropy?
    categorical_loss_scalar = 5.0
    categorical_loss = losses.categorical_crossentropy(gt[:, 1:], pred[:, 1:])

    total_loss = num_objects_loss_scalar * num_objects_loss + categorical_loss_scalar  * categorical_loss
        

    return total_loss

def custom_accuracy_2(gt, pred):
    return metrics.categorical_accuracy(gt[:, 1:], pred[:, 1:])

def custom_accuracy_1(gt, pred):
    # acc = K.sum(K.flatten(K.square(gt[:, 0] - pred[:, 0])))
    return metrics.mean_squared_error(gt[:, 0], pred[:, 0])


def customloss(gt, pred):
    """ Custom loss function to apply categorical cross entropy to the classification
    results and L2 loss to confidence and box coordinates"""
    #assert gt.shape == pred.shape
    
    
    f1 = lambda: tf.constant(0.5)
    f2 = lambda: tf.constant(0.5)
    
    scale_factor = tf.case([(tf.less(pred[batch,i,j,0], tf.constant(1.0)), f1)], default=f2)
    # Catgeorical crossentropy for classwise confidence
    loss_from_class_preds = K.categorical_crossentropy(gt[batch,i,j,5:], pred[batch,i,j,5:])
    loss_from_class_preds = tf.multiply(loss_from_class_preds, scale_factor)
    loss = tf.add(loss, loss_from_class_preds)
    # MSE for confidence and bounding box predictions
    bb_mse = K.mean(K.square(gt[i,j,:5] - pred[i,j,:5]))
    bb_mse = tf.multiply(bb_mse, scale_factor)
    loss = tf.add(loss , bb_mse)
    
    loss = K.variable(0.,dtype='float32')
#    print(gt.shape)
#    print(pred.shape)
#    try: 
#        batch_size = int(gt.shape[0])
#    except:
#        batch_size = 32
#        
        
        
#    for batch in range(batch_size):
#        for i in range(7):
#            for j in range(7):
#                 # Check if the box contains an object and adjust loss accordingly
#                scale_factor = tf.case([(tf.less(pred[batch,i,j,0], tf.constant(1.0)), f1)], default=f2)
#                # Catgeorical crossentropy for classwise confidence
#                loss_from_class_preds = K.categorical_crossentropy(gt[batch,i,j,5:], pred[batch,i,j,5:])
#                loss_from_class_preds = tf.multiply(loss_from_class_preds, scale_factor)
#                loss = tf.add(loss, loss_from_class_preds)
#                # MSE for confidence and bounding box predictions
#                bb_mse = K.mean(K.square(gt[i,j,:5] - pred[i,j,:5]))
#                bb_mse = tf.multiply(bb_mse, scale_factor)
#                loss = tf.add(loss , bb_mse)
            
    return tf.divide(loss , batch_size)