# Udacity ND013 Self-Driving Car

## Project 3 Behavioral Cloning Project
The model is based on the train steering model from Comma.ai research.
https://github.com/commaai/research
Training image is extracted to the focus area 70:130, 320 which trims image size to 60, 320. The focus area greatly improved training result and decreased training time.
Left and Right camera image is used to train recovery from off center. Left camera was set steering_angle - 0.1 and Right camera was set steering_angle + 0.1.
Driving steering angle is the opposite direction of training - steering_angle = -steering_angle.
Keras model.fit_generator is implement to allow training set: batch size=64.
Linux 50Hz simulator is used to sample large training set - 30k frames are used for training. Including Left and Right data, total training data is 100k.

## To do
More training data of curve track is needed.

## Credits
Thanks my Udacity mentor Maxime Leclerc for the great support!
