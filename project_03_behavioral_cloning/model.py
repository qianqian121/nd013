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

def load_steering_coded():
    nb_classes = 13
    steering = pd.read_csv('driving_log.csv', header=None, usecols=[3])
    steering = steering[3].tolist()

    steer_code = []
    for s in steering:
      code = 0
      if s < -0.45:
        code = 0
      elif s < -0.35:
        code = 0
      elif s < -0.25:
        code = 1
      elif s < -0.175:
        code = 2
      elif s < -0.125:
        code = 3
      elif s < -0.075:
        code = 4
      elif s < -0.025:
        code = 5
      elif s < 0.025:
        code = 6
      elif s < 0.075:
        code = 7
      elif s < 0.125:
        code = 8
      elif s < 0.175:
        code = 9
      elif s < 0.25:
        code = 10
      elif s < 0.35:
        code = 11
      else:
        code = 12
      steer_code.append(code)

    #steerings.extend(steering)
    #print(steerings)
    steer_array = np.asarray(steer_code)
    np.savetxt('steer_code.txt', steer_code, fmt='%d')
    print(steer_array.shape)
    steer_array = np_utils.to_categorical(steer_array, nb_classes)
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
        # print(steering[i])
        # print(speed[i])
        if speed[i] < 0.1:
            steering_speed.append(0)
        else:
            # steering_speed.append(steering[i] * speed[i])
            steering_speed.append(steering[i] * 25.0)

    print("--------------")
    steer_array = np.asarray(steering_speed)
    print(steer_array.shape)
    return steer_array
    #print(steerings[:100])

def load_steering_speed_three():
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

def load_data_trim():
    #img_path = 'test'
    img_path = 'IMG'
    imglist = os.listdir(img_path)
    list.sort(imglist)
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
        npimg = np.array(img)
        npimg = npimg[70:, :, :]
        images.append(npimg)

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

def get_model_code(time_len=1):
    # ch, row, col = 3, 160, 320  # camera format
    ch, row, col = 3, 90, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(13))

    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model

def get_model(time_len=1):
    # ch, row, col = 3, 160, 320  # camera format
    ch, row, col = 3, 90, 320  # camera format

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
    model.add(Dense(13))

    # model.compile(optimizer="adam", loss="mse")
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

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
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
    parser.add_argument('--port', type=int, default=5557, help='Port of server.')
    parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
    parser.add_argument('--batch', type=int, default=64, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
    parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
    parser.set_defaults(skipvalidate=False)
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()

    print(args.epoch)

    y_train = load_steering_coded()
    # y_train = load_steering()
    x_train = load_data_trim()
    #y_train = y_train[:4]
    #save_pickle(x_data, y_data)
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25)
    print(x_train.shape, y_train.shape)
    # print(x_valid.shape, y_valid.shape)
    print("Starting model weights and configuration file.")

    model = get_model()
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        nb_epoch=args.epoch,
        verbose=1,
        validation_split=0.1,
        # validation_data=(x_valid, y_valid),
        shuffle=True
    )
    print(history.history.keys())
    print("Saving model weights and configuration file.")

    # if not os.path.exists("./outputs/steering_model"):
    #     os.makedirs("./outputs/steering_model")

    model.save_weights("model.h5", True)
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
