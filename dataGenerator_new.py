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
import random

class DataGenerator_new(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=4, dim=(224,224,3),
                 n_classes=20, shuffle=True, img_dir="", label_dir="", train=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.train = train
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

    # flip an image for training
    def _flip(self,img):
        return np.flip(img, axis=1)

    # rotate image for training
    def _rotate(self,img):
        theta = random.randint(-30,30)
        rows,cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        out = cv2.warpAffine(img,M,(cols,rows))
        return out

    # shift an image vertically or horizontally
    def _shift(self, img):
        hshift = random.randint(-20, 20)
        vshift = random.randint(-20, 20)
        img = np.roll(img, hshift, axis=1)
        img = np.roll(img, vshift, axis=0)
        return img

    def __data_generation(self, list_IDs_temp, list_labels_temp):
        N_LEN = 1
        C_LEN = 20
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty([self.batch_size,self.dim[0],self.dim[1],self.dim[2]])
        y = np.empty([self.batch_size, N_LEN + C_LEN])

        # Generate data
        BASE_DIR = "M:/Documents/Courses/CSE586/finalProject/CV2_final_project"
        IMG_DIR = self.img_dir
        LABEL_DIR = self.label_dir
        avg_colors = (124.59085262283071, 124.91715538568295, 124.90344722141644) # precomputed on the training set in bgr order
        for i, ID in enumerate(list_IDs_temp):
            image_path = os.path.join(IMG_DIR, ID)
            label_path = os.path.join(LABEL_DIR, list_labels_temp[i])

            # load image
            img = cv2.imread(image_path)
            
            # apply augmentation
            # 50% chance to flip horizontally
            if self.train and random.randint(0,1):
                img = self._flip(img)
            #rotate +/- 30 degrees
            if self.train and random.randint(0, 1):
                img = self._rotate(img)
            if self.train and random.randint(0, 1):
                img = self._shift(img)

            # resize image and remove average colors
            img = cv2.resize(img, self.dim[0:2])
            img = img - avg_colors
            X[i,] = img

            # Store class
            y[i,] = np.load(label_path)
            
        return X, y