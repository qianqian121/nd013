#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Adam

import time
from PIL import Image
from PIL import ImageOps

# SKLEARN
import sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np

import pandas as pd
#import pickle
import dill

def save_pickle(train_features, train_labels):
    # Save the data for easy access
    pickle_file = 'train.pickle'
    if True:
        print('Saving data to pickle file...')
        try:
            with open('train.pickle', 'wb') as pfile:
                # d = klepto.archives.dir_archive('data', cached=True, serialized=True)
                # d['train_dataset'] = train_features
                # d['train_labels'] = train_labels
                # d.dump()
                # d.clear()
                dill.dump(
                # pickle.dump(
                    {
                        'train_dataset': train_features,
                        'train_labels': train_labels,
                    },
                    pfile)
                    #pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')

def get_model_comma_ai(time_len=1):
# def get_model_comma_ai(time_len=1):
#     ch, row, col = 3, 80, 320  # camera format
    ch, row, col = 3, 60, 320  # camera format
    # ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    # model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    adam = Adam(lr=0.0001)
    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer=adam,
                  loss='mse',
                  metrics=['accuracy'])

    return model

# Nvidia
# def get_model(time_len=1):
def get_model(time_len=1):
    # ch, row, col = 3, 80, 320  # camera format
    ch, row, col = 3, 66, 200  # camera format
    # ch, row, col = 3, 160, 200  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Convolution2D(64, 3, 3))
    model.add(Flatten())
    model.add(Dense((1164), activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1, name='output'))

    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='mse',
                  metrics=['accuracy'])

    return model

def get_img_name(imglist):
    # print(imglist)
    imgnames = []
    for p in imglist:
        # print(p)
        index = p.rfind('/')
        # print(index)
        # print(p[index:])
        imgnames.append(p[index+1:])
    return imgnames

def test_gen():
    # img_log = pd.read_csv('driving_log.csv',header=None, dtype={0:str}, usecols=[0])
    # img_log = pd.read_csv('driving_log.csv', header=None, dtype=object, usecols=[0,1,2])
    driving_log = pd.read_csv('driving_log.csv', header=None, usecols=[0,1,2,3, 6])
    center = driving_log[0].astype(str).tolist()
    center = get_img_name(center)
    left = driving_log[1].astype(str).tolist()
    left = get_img_name(left)
    right = driving_log[2].astype(str).tolist()
    right = get_img_name(right)
    steering = driving_log[3].tolist()
    speed = driving_log[6].tolist()
    imglist = []

    for i in range(len(steering)):
        if (abs(steering[i]) > 0.2):
            tup = (center[i], steering[i], 'c')
            imglist.append(tup)
            tup = (center[i], steering[i], 'c')
            imglist.append(tup)

    for i in range(len(steering)):
        tup = (center[i], steering[i], 'c')
        imglist.append(tup)
        tup = (center[i], -steering[i], 'm')
        imglist.append(tup)
        tup = (left[i], steering[i]+0.25, 'l')
        imglist.append(tup)
        tup = (right[i], steering[i]-0.25, 'r')
        imglist.append(tup)
    # print(imglist)
    return imglist

import random
SEED = 323

import cv2
batch_num = 0

def data_gen(imglist, batchsize):
    global batch_num
    imgpath = 'IMG'
    batch = 0
    images = []
    angles = []
    while 1:
        random.shuffle(imglist)
        # print(imglist)
        for tup in imglist:
            # print(tup)
            img = Image.open(imgpath + '/' + tup[0])
            # print(imgpath + '/' + tup[0])
            # img.load()
            # img = img.getdata()
            # print(img)
            angle = np.array([tup[1]])
            if (tup[2] == 'm'):
                # img.save('center0.png')
                img = ImageOps.mirror(img)
                # img.save('mirro0.png')
            image_array = np.asarray(img)

            # print(image_array)
            # print(image_array.shape)
            # image_array = image_array[70:130, :, :]
            image_array = image_array[50:140, :, :]
            image_array = cv2.resize(image_array, (200, 66), interpolation = cv2.INTER_AREA)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)

            images.append(image_array)
            angles.append(angle)
            # transformed_image_array = image_array[None, :, :, :]
            # transformed_image_array = image_array[None, 70:130, :, :]
            # transformed_image_array = image_array[None, 60:140, :, :]
            # yield _x_train, _y_train

            # print(angle.shape)
            batch += 1
            if (batch % batchsize) == 0:
                # print('...batch...')
                # print(batch)
                # batch = 0
                batch_num = batch_num + 1
                X_train = np.array(images)
                y_train = np.array(angles)
                images = []
                angles = []
                yield sklearn.utils.shuffle(X_train, y_train)
                # yield X_train, y_train
                # yield transformed_image_array, angle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=2, help='Number of epochs.')

    parser.set_defaults(skipvalidate=False)
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()

    print(args.epoch)

    print("Starting model weights and configuration file.")

    model = get_model()
    samples = 16384
    batch = 128
    history = model.fit_generator(
        # data_gen(test_gen(), 1),
        # samples_per_epoch=1,
        data_gen(test_gen(), batch),
        # samples_per_epoch=2048,
        # samples_per_epoch=512,
        samples_per_epoch = samples,
        # samples_per_epoch=100000,
        nb_epoch=args.epoch,
        # validation_data=gen(20, args.host, port=args.val_port),
        nb_val_samples=2560,   #validation sample size
        max_q_size=65536
    )

    print(batch_num)
    print(history.history.keys())
    print("Saving model weights and configuration file.")

    # if not os.path.exists("./outputs/steering_model"):
    #     os.makedirs("./outputs/steering_model")

    model.save_weights("model.h5", True)
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
