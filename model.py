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
from keras import optimizers
from keras.utils import np_utils


image_IDs = os.listdir('C:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/')[:15000]
labels = os.listdir('C:/VOC_individual_np/labels')[:15000]
val_IDs = os.listdir('C:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/')[15000:]
val_labels = os.listdir('C:/VOC_individual_np/labels')[15000:]
conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
for layer in conv_base.layers[:14]:
    layer.trainable = False
    
# BB regression
input = Input(shape=(224,224,3))
conv_base = conv_base(input)
#conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(224, 224, 3))(input)
x = Flatten()(conv_base)
x = Dense(512, activation='relu')(x)
probs = Dense(20, activation='softmax')(x)
num_objs = Dense(1)(x)
out = concatenate([num_objs,probs])

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



adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss=SkyUtils.custom_loss_2, optimizer=adam, metrics=[SkyUtils.custom_accuracy_1, SkyUtils.custom_accuracy_2])

training_generator = dataGenerator.DataGenerator(image_IDs, labels)
validation_generator = dataGenerator.DataGenerator(val_IDs, val_labels)

model.summary()
epochs = 25
custom_acc1 = []
custom_acc2 = []
loss = []
val_custom_acc1 = []
val_custom_acc2 = []
val_loss = []
for i in range(epochs):
    history = model.fit_generator(generator = training_generator, use_multiprocessing = False, workers = 16)
    val_history = model.evaluate_generator(generator = validation_generator, use_multiprocessing=False, workers = 16)
    custom_acc1.append(history.history['custom_accuracy_1'])
    custom_acc2.append(history.history['custom_accuracy_2'])
    loss.append(history.history['loss'])
    val_custom_acc1.append(val_history[1])
    val_custom_acc2.append(val_history[2])
    val_loss.append(val_history[0])
    # plot
    x = [x for x in range(len(loss))]
    pylab.plot(x,loss,'r-',label='loss')
    pylab.plot(x,val_loss,'b-',label='validation loss')
    pylab.legend(loc='upper left')
    pylab.show()
model.save('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_{}.h5'.format(i))
model.save_weights('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_weights_{}.h5'.format(i))

print('making all layers trainable...')
for layer in model.layers[1].layers:
    layer.trainable = True
model.summary()
for i in range(epochs):
    history = model.fit_generator(generator = training_generator, use_multiprocessing = False, workers = 16)
    val_history = model.evaluate_generator(generator = validation_generator, use_multiprocessing=False, workers = 16)
    custom_acc1.append(history.history['custom_accuracy_1'])
    custom_acc2.append(history.history['custom_accuracy_2'])
    loss.append(history.history['loss'])
    val_custom_acc1.append(val_history[1])
    val_custom_acc2.append(val_history[2])
    val_loss.append(val_history[0])
model.save('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_{}.h5'.format(i))
model.save_weights('C:/Users/ander/OneDrive/Desktop/DL/CV2_final_project/model_saves/first_model_weights_{}.h5'.format(i))

from matplotlib import pyplot as plt
import pylab
x = [x for x in range(len(loss))]
pylab.plot(x,loss,'r-',label='loss')
pylab.plot(x,val_loss,'b-',label='validation loss')
pylab.legend(loc='upper left')
pylab.show()

plt.plot(x,loss,'r-',x,val_loss,'b-',label='loss')
plt.plot(x,custom_acc1,x,val_custom_acc1,label='custom_acc1')
plt.plot(x,custom_acc2,x,val_custom_acc2,label='custom_acc2')
plt.legend()
IMG_DIR = 'C:/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'
LABEL_DIR = 'C:/VOC_individual_np/labels'
indices = [x for x in range(17000,17010)]
SkyUtils.makePredictions2(model,IMG_DIR,LABEL_DIR,indices)
img = np.load('D:/VOC_individual_np/images/1.npy')
gt = np.load('D:/VOC_individual_np/labels/1.npy')
#cv2.imshow('img',img);cv2.waitKey()
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


