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
prediction_indices = [random.randint(0, len(validation_image_IDs)-1) for x in range(20)]

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
# apply 4096 hidden layer
x = Dense(HIDDEN_SIZE, activation='relu')(x)
x = Dropout(rate=0.2)(x)
# get confidence scores and reshape
num_images = Dense(N_LEN)(x)
class_probs = Dense(C_LEN, activation="softmax")(x)
out = concatenate([num_images, class_probs],axis=1)
print(out)

print("creating model ...")
model = Model(inputs=input, outputs=out)
# compile model or whatever
adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
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

learning_rate_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    verbose=1,
    mode="min",
    epsilon=0.001,
    cooldown=0,
    min_lr=0
)

# make predictions on the test data
SkyUtils.makePredictions2(model=model, image_dir=VALIDATION_IMG_DIR, label_dir=VALIDATION_LABEL_DIR, indices=prediction_indices)

print("training ...")
epochs = 20
train_total_loss = []
train_num_images_loss = []
train_top_1_acc_loss = []
train_top_5_acc_loss = []
train_top_10_acc_loss = []
val_total_loss = []
val_num_images_loss = []
val_top_1_acc_loss = []
val_top_5_acc_loss = []
val_top_10_acc_loss = []
for i in range(epochs):
    history = model.fit_generator(
        generator=training_generator,
        use_multiprocessing=MULTIPROCESSING,
        workers=PARALLEL
    )
    print(history.history)
    scores = model.evaluate_generator(validation_generator)

    train_total_loss.append(history.history['loss'])
    train_num_images_loss.append(history.history['custom_accuracy_num'])
    train_top_1_acc_loss.append(history.history['custom_accuracy_top_1'])
    train_top_5_acc_loss.append(history.history['custom_accuracy_top_5'])
    train_top_10_acc_loss.append(history.history['custom_accuracy_top_10'])
    
    val_total_loss.append(scores[0])
    val_num_images_loss.append(scores[1])
    val_top_1_acc_loss.append(scores[2])
    val_top_5_acc_loss.append(scores[3])
    val_top_10_acc_loss.append(scores[4])

    print("Validation: ")
    print("total loss: {}".format(scores[0]))
    print("top 1 accuracy: {}".format(scores[2]))
    print("top 5 accuracy: {}".format(scores[3]))
    print("top 10 accuracy: {}".format(scores[4]))

    model.save(os.path.join(BASE_DIR, "models/big_model_{}.h5".format(i)))
    model.save_weights(os.path.join(BASE_DIR, "models/big_model_weights_{}.h5".format(i)))

    # make predictions on the test data
    SkyUtils.makePredictions2(model=model, image_dir=VALIDATION_IMG_DIR, label_dir=VALIDATION_LABEL_DIR, indices=prediction_indices)

# plot some results
epoch_array = [x for x in range(epochs)]
plt.plot(epoch_array, train_top_1_acc_loss, label="train top 1")
plt.plot(epoch_array, train_top_5_acc_loss, label="train top 5")
plt.plot(epoch_array, train_top_10_acc_loss, label="train top 10")
plt.plot(epoch_array, val_top_1_acc_loss, label="val top 1")
plt.plot(epoch_array, val_top_5_acc_loss, label="val top 5")
plt.plot(epoch_array, val_top_10_acc_loss, label="val top 10")
plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(epoch_array, train_total_loss, label="train total loss")
plt.plot(epoch_array, train_num_images_loss, label="train num loss")
plt.plot(epoch_array, val_total_loss, label="val total loss")
plt.plot(epoch_array, val_num_images_loss, label="val num loss")
plt.legend(loc="best")
plt.show()

# testing out some predictions
# print("testing ...")
# img = np.load(os.path.join(BASE_DIR, "data/preprocessed/images/1.npy"))
# gt = np.load(os.path.join(PREPROCESSED_DIR, "labels/1.npy"))
# cv2.imshow('img',img);cv2.waitKey()

# prediction = model.predict(np.expand_dims(img, axis=0))
# print(prediction)
#canvas = cv2.imread(os.path.join(BASE_DIR, "data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg"))
#canvas = cv2.resize(canvas, (IMG_SIZE[0], IMG_SIZE[1]))
#output = SkyUtils.drawRectsFromPred(prediction[0],canvas)
#cv2.imshow('output', output); cv2.waitKey()

print("saving model ...")
model.save(os.path.join(BASE_DIR, "models/big_model.h5"))
model.save_weights(os.path.join(BASE_DIR, "models/big_model_weights.h5"))