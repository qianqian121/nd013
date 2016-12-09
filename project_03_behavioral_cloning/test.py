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

import time
from PIL import Image
from PIL import ImageOps

# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import numpy as np

import pandas as pd
#import pickle
import dill
#import cPickle as pickle
# import klepto

from keras.models import model_from_json

# from numpy import *
def load_steering():
    steering = pd.read_csv('driving_log.csv', header=None, usecols=[3])
    steering = steering[3].tolist()

    #steerings.extend(steering)
    #print(steerings)
    steer_array = np.asarray(steering)
    print(steer_array.shape)
    return steer_array
    #print(steerings[:100])

def load_steering_three():
    steering = pd.read_csv('driving_log.csv', header=None, usecols=[3])
    steering = steering[3].tolist()
    steerings = []
    for i in range(3):
        steerings.extend(steering)
    #steerings.extend(steering)
    #print(steerings)
    steer_array = np.asarray(steerings)
    print(steer_array.shape)
    return steer_array
    #print(steerings[:100])


def load_steering_speed():
    driving_log = pd.read_csv('driving_log.csv', header=None, usecols=[3,6])
    steering = driving_log[3].tolist()
    speed = driving_log[6].tolist()

    steering_speed = []
    for i in range(len(steering)):
        print(steering[i])
        print(speed[i])
        if speed[i] < 0.1:
            steering_speed.append(0)
        else:
            steering_speed.append(steering[i] * speed[i])

    print("--------------")
    print(steering_speed)
    steering_speeds = []
    for i in range(3):
        steering_speeds.extend(steering_speed)
    #steerings.extend(steering)
    #print(steerings)
    steer_array = np.asarray(steering_speeds)
    print(steer_array.shape)
    return steer_array
    #print(steerings[:100])

def load_data():
    #img_path = 'test'
    img_path = 'IMG'
    imglist = os.listdir(img_path)
    list.sort(imglist)

    with open('img_files.txt', 'w') as fp:
        for s in imglist:
            fp.write("%s\n" % s)
    # create matrix to store all flattened images
    # immatrix = array([array(Image.open('IMG' + '/' + im)).flatten()
    #                  for im in imglist], 'f')
    # import glob
    # cv_img = []
    # for img in glob.glob("Path/to/dir/*.jpg"):
    #  n = cv2.imread(img)
    #  cv_img.append(n)
    images = []
    for imgfile in imglist:
        img = Image.open(img_path + '/' + imgfile)
        #img = load_img(img_path + '/' + imgfile)
        #img = img_to_array(img)
        #images.append(img)
        # gray = img.convert('L')
        images.append(np.array(img))

    immatrix = np.asarray(images, dtype=np.uint8)
    print(immatrix.shape)
    return immatrix

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

def get_model(time_len=1):
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
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

    model.compile(optimizer="adam", loss="mse")

    return model

def get_model_new(time_len=1):
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    #y_train = load_steering_speed()
    y_train = load_steering()
    x_train = load_data()

    print("Starting model weights and configuration file.")

    result_train = model.predict(x_train, batch_size=64, verbose=1)
    print("Finished model weights and configuration file.")

    np.savetxt("steering.txt", y_train, delimiter=",")
    np.savetxt("result.txt", result_train, delimiter=",")
