# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:32:04 2018

@author: ander
"""

import numpy as np
import cv2
import os
import dataGenerator
import SkyUtils
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.datasets import cifar100
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense,
from keras.layers import Dropout, Flatten, BatchNormalization, Activation, 
from keras.layers import Reshape, concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils

BASE_DIR = "M:\Documents\Courses\CSE586\finalProject\CV2_final_project"
S = 7
B = 1
C_LEN = 20
IMG_SIZE = (224, 224, 3)

# get the current activate directory
os.chdir(BASE_DIR)

image_IDs = os.listdir(os.path.join(BASE_DIR, "data/preprocessed/images"))
image_IDs = os.listdir(os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages/"))
labels = os.listdir('D:/VOC_individual_np/labels')
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=IMG_SIZE)

# BB regression
input = Input(shape=IMG_SIZE
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=IMG_SIZE)(input)
x = Flatten()(conv_base)
x = Dense(4096, activation='relu')(x)
conf = Dense(S*S*1, activation='sigmoid')(x)
conf = Reshape((S,S,1))(conf)
bb = Dense(S*S*4)(x)
bb = Reshape((S,S,4))(bb)
classes = Dense(S*S*C_LEN, activation='softmax')(x)
classes = Reshape((S,S,C_LEN))(classes)
out = concatenate([conf,bb,classes],axis=3)

model = Model(inputs=input, outputs=out)

#
#model = Sequential()
#model.add(conv_base)
##model.add(Convolution2D(25, (1,1)))
#model.add(Flatten())
#model.add(Dense(4096, activation='sigmoid'))
#model.add(Dropout(0.5))
#model.add(Dense(S * S * 25))
#model.add(Reshape((S,S,25)))

for layer in conv_base.layers:
    layer.trainable = False
model.compile(loss=SkyUtils.customloss, optimizer='adam', metrics=['accuracy'])

training_generator = dataGenerator.DataGenerator(image_IDs, labels)

epochs = 100
acc = []
loss = []
for i in range(epochs):
    history = model.fit_generator(generator = training_generator, use_multiprocessing = False, workers = 4)
    acc.append(history.history['acc'])
    loss.append(history.history['loss'])
    model.save(os.path.join(BASE_DIR, "modelS/first_model_{}.h5".format(i)))
    model.save_weights(os.path.join(BASE_DIR, "models/first_model_weights_{}.h5".format(i)))
    
img = np.load(os.path.join(BASE_DIR, "data/preprocessed/images/1.npy"))
gt = np.load(os.path.join(BASE_DIR, "data/preprocessed/labels/1.npy"))
cv2.imshow('img',img);cv2.waitKey()

prediction = model.predict(np.expand_dims(img, axis=0))
canvas = cv2.imread(os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"))
canvas = cv2.resize(canvas, (IMG_SIZE[0], IMG_SIZE[1]))
output = SkyUtils.drawRectsFromPred(prediction[0],canvas)
cv2.imshow('output', output); cv2.waitKey()

model.save(os.path.join(BASE_DIR, "models/first_model.h5"))
model.save_weights(os.path.join(BASE_DIR, "models/first_model_weights.h5"))

from time import time

