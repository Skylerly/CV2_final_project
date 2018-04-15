# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:32:04 2018

@author: ander
"""

import numpy as np
import cv2
import os
os.chdir('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project')
import dataGenerator
import SkyUtils
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.datasets import cifar100
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation, Reshape, concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils

image_IDs = os.listdir('D:/VOC_individual_np/images')
image_IDs = os.listdir('D:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/')
labels = os.listdir('D:/VOC_individual_np/labels')
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
for layer in conv_base.layers:
    layer.trainable = False
    
# BB regression
input = Input(shape=(224,224,3))
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))(input)
x = Flatten()(conv_base)
x = Dense(4096, activation='relu')(x)
conf = Dense(7*7*1, activation='sigmoid')(x)
conf = Reshape((7,7,1))(conf)
bb = Dense(7*7*4)(x)
bb = Reshape((7,7,4))(bb)
classes = Dense(7*7*20, activation='softmax')(x)
classes = Reshape((7,7,20))(classes)
out = concatenate([conf,bb,classes],axis=3)

model = Model(inputs=input, outputs=out)

#
#model = Sequential()
#model.add(conv_base)
##model.add(Convolution2D(25, (1,1)))
#model.add(Flatten())
#model.add(Dense(4096, activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(7 * 7 * 25))
#model.add(Reshape((7,7,25)))



model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

training_generator = dataGenerator.DataGenerator(image_IDs, labels)


epochs = 1
acc = []
loss = []
for i in range(epochs):
    history = model.fit_generator(generator = training_generator, use_multiprocessing = False, workers = 16)
    acc.append(history.history['acc'])
    loss.append(history.history['loss'])
    model.save('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_{}.h5'.format(i))
    model.save_weights('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_weights_{}.h5'.format(i))
    
img = np.load('D:/VOC_individual_np/images/1.npy')
gt = np.load('D:/VOC_individual_np/labels/1.npy')
cv2.imshow('img',img);cv2.waitKey()

prediction = model.predict(np.expand_dims(img, axis=0))
canvas = cv2.imread('D:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg')
canvas = cv2.resize(canvas, (224,224))
output = SkyUtils.drawRectsFromPred(prediction[0],canvas)
cv2.imshow('output', output); cv2.waitKey()

img = cv2.imread('D:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg')
img = cv2.resize(img, (224,224))
output = SkyUtils.overlayPredsByColor(prediction[0], img)
cv2.imshow('predictions by color',output);cv2.waitKey()

model.save('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model.h5')
model.save_weights('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_weights.h5')


