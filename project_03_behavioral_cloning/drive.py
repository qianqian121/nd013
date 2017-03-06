import argparse
import base64

from datetime import datetime
import os

import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import cv2

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    # image = image.resize((200, 160), Image.ANTIALIAS)
    image_array = np.asarray(image)

    image_array = image_array[50:140, :, :]
    image_array = cv2.resize(image_array, (200, 66), interpolation=cv2.INTER_AREA)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
    # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # convert numpy array data back to png image
    # img = Image.fromarray(image_array, 'RGB')
    # img.save('drive0.png')

    # transformed_image_array = image_array[None, 60:140, :, :]
    # transformed_image_array = image_array[None, 70:130, :, :]
    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    if False:
        steering_res = model.predict(transformed_image_array, batch_size=1)
        steering_code = steering_res.argmax(1)
        steer_dict = [-0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
        steering_angle = steer_dict[steering_code]
    # steering_angle *= 1.5
    # steering_angle = -steering_angle
    # print(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.2
    speed = float(speed)
    if False:
        if speed < 0.2:
            steering_angle = 0
        else:
            # steering_angle = steering_angle / speed
            steering_angle = steering_angle / 57.2958
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

    # save frame
    if args.image_folder != '':
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(args.image_folder, timestamp)
        image.save('{}.jpg'.format(image_filename))

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')

    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )

    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)