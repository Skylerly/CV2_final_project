# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:32:04 2018

@author: ander
"""
import numpy as np
import cv2
import os
import dataGenerator_new
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
from keras import optimizers
import tqdm
import matplotlib.pyplot as plt
import random

BASE_DIR = "M:/Documents/Courses/CSE586/finalProject/CV2_final_project"
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data/preprocessed_2")
TRAIN_IMG_DIR = os.path.join(PREPROCESSED_DIR, "train_images")
TRAIN_LABEL_DIR = os.path.join(PREPROCESSED_DIR, "train_labels")
VALIDATION_IMG_DIR = os.path.join(PREPROCESSED_DIR, "validation_images")
VALIDATION_LABEL_DIR = os.path.join(PREPROCESSED_DIR, "validation_labels")
N_LEN = 1
C_LEN = 20
IMG_SIZE = (224, 224, 3)
HIDDEN_SIZE = 1024
MULTIPROCESSING = False
PARALLEL = 1
BATCH_SIZE = 16

# get the current activate directory
os.chdir(BASE_DIR)

# loading all image and label filenames
train_image_IDs = os.listdir(TRAIN_IMG_DIR)
train_label_IDs = os.listdir(TRAIN_LABEL_DIR)
validation_image_IDs = os.listdir(VALIDATION_IMG_DIR)
validation_label_IDs = os.listdir(VALIDATION_LABEL_DIR)

# create random indices to make predictions for
prediction_indices = [random.randint(0, len(validation_image_IDs)-1) for x in range(30)]

print("loading images and labels and conv net ...")
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
x = Dropout(rate=0.4)(x)
# apply 4096 hidden layer
x = Dense(HIDDEN_SIZE, activation='relu')(x)
x = Dropout(rate=0.4)(x)
# get confidence scores and reshape
num_images = Dense(N_LEN)(x)
class_probs = Dense(C_LEN, activation="softmax")(x)
out = concatenate([num_images, class_probs],axis=1)

print("creating model ...")
model = Model(inputs=input, outputs=out)
model.load_weights("models/big_model_weights_7.h5")
# compile model or whatever
adam = optimizers.adam(lr=0.000001, beta_1=0.9, beta_2=0.999)
model.compile(
    loss=SkyUtils.custom_loss_2,
    optimizer=adam,
    metrics=[
        SkyUtils.custom_accuracy_num,
        SkyUtils.custom_accuracy_top_1,
        SkyUtils.custom_accuracy_top_5,
        SkyUtils.custom_accuracy_top_10
    ]
)
# create the data generator to use
training_generator = dataGenerator_new.DataGenerator_new(train_image_IDs, train_label_IDs, batch_size=BATCH_SIZE,
    dim=IMG_SIZE, n_classes=C_LEN, shuffle=True, img_dir=TRAIN_IMG_DIR, label_dir=TRAIN_LABEL_DIR, train=True)
validation_generator = dataGenerator_new.DataGenerator_new(validation_image_IDs, validation_label_IDs, batch_size=BATCH_SIZE,
    dim=IMG_SIZE, n_classes=C_LEN, shuffle=True, img_dir=VALIDATION_IMG_DIR, label_dir=VALIDATION_LABEL_DIR, train=False)

# make predictions on the test data
SkyUtils.makePredictions2(model=model, image_dir=VALIDATION_IMG_DIR, label_dir=VALIDATION_LABEL_DIR, indices=prediction_indices)

scores = model.evaluate_generator(validation_generator)

print(scores)
print("Validation Scores: ")
print("total loss: {}".format(scores[0]))
print("top 1 accuracy: {}".format(scores[2]))
print("top 5 accuracy: {}".format(scores[3]))
print("top 10 accuracy: {}".format(scores[4]))

# # plot some results
# epoch_array = [x for x in range(epochs)]
# plt.plot(epoch_array, train_top_1_acc_loss, label="train top 1")
# plt.plot(epoch_array, train_top_5_acc_loss, label="train top 5")
# plt.plot(epoch_array, train_top_10_acc_loss, label="train top 10")
# plt.plot(epoch_array, val_top_1_acc_loss, label="val top 1")
# plt.plot(epoch_array, val_top_5_acc_loss, label="val top 5")
# plt.plot(epoch_array, val_top_10_acc_loss, label="val top 10")
# plt.legend(loc="best")
# plt.show()

# plt.figure()
# plt.plot(epoch_array, train_total_loss, label="train total loss")
# plt.plot(epoch_array, train_num_images_loss, label="train num loss")
# plt.plot(epoch_array, val_total_loss, label="val total loss")
# plt.plot(epoch_array, val_num_images_loss, label="val num loss")
# plt.legend(loc="best")
# plt.show()