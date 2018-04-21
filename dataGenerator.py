# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:52:32 2018

@author: ander

NOTE: Original code taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html and modified
"""

import numpy as np
import keras
import cv2
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=4, dim=(224,224,3),
                 n_classes=10, shuffle=True, ):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp,list_labels_temp):
        B = 1
        S = 7
        C_LEN = 20
        IMG_SIZE = (224, 224)
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size,self.dim[0],self.dim[1],self.dim[2]])
        y = np.empty([self.batch_size, S, S, (5*B+C_LEN)])

        # Generate data
        BASE_DIR = "M:/Documents/Courses/CSE586/finalProject/CV2_final_project"
        IMG_DIR = os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages/")
        LABELS_DIR = os.path.join(BASE_DIR, "data/preprocessed/labels/")
        avg_colors = (124.59085262283071, 124.91715538568295, 124.90344722141644) # precomputed on the training set in bgr order
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(IMG_DIR + ID)
            img = cv2.resize(img, IMG_SIZE)
            img = img - avg_colors
            #X[i,] = np.load('D:/VOC_individual_np/images/' + ID)
            X[i,] = img

            # Store class
            #TODO: modify this to transform our training labels too
            y[i,] = np.load(LABELS_DIR + list_labels_temp[i])
        
        print("X: {}".format(X.shape))
        print("y: {}".format(y.shape))

        return X, y