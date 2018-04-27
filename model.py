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
import yololoss_implementation1
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.datasets import cifar100
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense
from keras.layers import Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Reshape, concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
import tqdm

BASE_DIR = "M:/Documents/Courses/CSE586/finalProject/CV2_final_project"
S = 7
B = 1
C_LEN = 20
IMG_SIZE = (224, 224, 3)
# HIDDEN_SIZE = 4096
HIDDEN_SIZE = 256
MULTIPROCESSING = False
PARALLEL = 1

# get the current activate directory
os.chdir(BASE_DIR)

print("loading images and labels and conv net ...")
image_IDs = os.listdir(os.path.join(BASE_DIR, "data/preprocessed/images"))
image_IDs = os.listdir(os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages/"))
labels = os.listdir(os.path.join(BASE_DIR, "data/preprocessed/labels"))
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=IMG_SIZE)

# set weights to not trainable
for layer in conv_base.layers:
    layer.trainable = False

print("creating model structure ...")
# input image
input = Input(shape=IMG_SIZE)
# put image through pretrained network
base = conv_base(input)
# flatten for dense layers
x = Flatten()(base)
# apply 4096 hidden layer
x = Dense(HIDDEN_SIZE, activation='relu')(x)
# get confidence scores and reshape
conf = Dense(S*S, activation='sigmoid')(x)
conf = Reshape((S,S,1))(conf)
# get bounding boxes and reshape
bb = Dense(S*S*4)(x)
bb = Reshape((S,S,4))(bb)
# get class predictions and reshape
classes = Dense(S*S*C_LEN, activation='softmax')(x)
classes = Reshape((S,S,C_LEN))(classes)
# concatenate the layers for the output
out = concatenate([conf,bb,classes],axis=3)
print(out)

# prediction output: (confidence, x, w, y, h, probs)


print("creating model ...")
model = Model(inputs=input, outputs=out)
# compile model or whatever
model.compile(loss=SkyUtils.customloss, optimizer='adam', metrics=['accuracy'])
# create the data generator to use
training_generator = dataGenerator.DataGenerator(image_IDs, labels)

print("training ...")
epochs = 100
acc = []
loss = []
for i in range(epochs):
    history = model.fit_generator(
        generator=training_generator,
        use_multiprocessing=MULTIPROCESSING, 
        workers=PARALLEL
    )
    acc.append(history.history['acc'])
    loss.append(history.history['loss'])
    print("epoch {} / {}, loss: {}".format(i, epochs, loss[-1]))
    print("saving model ...")
    model.save(os.path.join(BASE_DIR, "modelS/first_model_{}.h5".format(i)))
    model.save_weights(os.path.join(BASE_DIR, "models/first_model_weights_{}.h5".format(i)))

# testing out some predictions
print("testing ...")
img = np.load(os.path.join(BASE_DIR, "data/preprocessed/images/1.npy"))
gt = np.load(os.path.join(BASE_DIR, "data/preprocessed/labels/1.npy"))
cv2.imshow('img',img);cv2.waitKey()

prediction = model.predict(np.expand_dims(img, axis=0))
canvas = cv2.imread(os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"))
canvas = cv2.resize(canvas, (IMG_SIZE[0], IMG_SIZE[1]))
output = SkyUtils.drawRectsFromPred(prediction[0],canvas)
cv2.imshow('output', output); cv2.waitKey()

print("saving model ...")

model.save(os.path.join(BASE_DIR, "models/first_model.h5"))
model.save_weights(os.path.join(BASE_DIR, "models/first_model_weights.h5"))