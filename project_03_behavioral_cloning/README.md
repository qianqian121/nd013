#**Behavioral Cloning** 

Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.jpg "Model Visualization"
[image2]: ./examples/center.jpg "Center"
[image3]: ./examples/offroad.jpg "off road"
[image4]: ./examples/side_camera1.jpg "side camera Image"
[image5]: ./examples/side_camera2.jpg "side camera Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.json containing a neural network model
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 for the tracking self driving video

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. 
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network from Nvidia autopilot demo.

####2. Attempts to reduce overfitting in the model

The training output - loss and accuracy is very different from MNIST and traffic sign projects. I noticed that loss can keep decreasing but accuracy will keep around 0.25f for my model. But the vehicle drives ok on the track.

But I did realize that for a normal driving around the track, the straight track frames is much more than the curves. This will cause the model overfit for the straight line images. I added more frames to the training set if abs(steering_angle) is larger than 0.2. Adding more curve data sets to ensure that the model was not overfitting (model.py line 142-147). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, the learning rate was tuned manually from default 0.001 to 0.0001 (model.py line 25). Smaller learning rate to make sure the model is not overfitting.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. When I first worked on the project, I used keyboard to control the vehicle and record data. But keyboard does not generate smooth steering angel. I used XBOX controller to collect data which worked really well. However, to record recovering from the sides of the road is really challenging - take lots of time and wrong driving operations will damage the whole data set. I plan to write some code later to record a temporary video clip, if the clip replay is good for recovering, then convert the video to the training data sets.

My models works okay with center data and recovering data. I didn't add the second track data.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first attempt was to use a convolution neural network model similar to the comma.ai auto steering model. Because it's small enough to be able to be trained on a less powerful PC such as laptop.

Comma.ai can be trained really fast, but I found it was not able to give me the consistent result. I realized it may require higher quality data set to finish the track.
 
 I then switched to Nvidia CNN model. But the bigger model takes more time to train and also takes lots of time to tune the pipeline and parameters. I also spent lots of time working on different dockers to speed up the training.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:
 ###### The curve before the bridge
 ###### right before and on the bridge
 ###### The first curve after the bridge
 ###### The second curve after the bridge
 
 This project is really challenging and also has lots of fun.
 
 To improve the driving behavior in these cases, I spent lots of effort changing different model, recollecting the training data, add more data for the curve, tuning the training parameter, slowing down the throttle/speed, etc...

At the end of the process, the vehicle is able to drive autonomously around the track. Right before the bridge, there is one spot where the car hit the yellow line then recover back. Other than that, the vehicle can run multiple laps without leaving the road.

I plan to improve my model later to allow full speed finishing the first track, then finish the second track.

####2. Final Model Architecture

The final model architecture (model.py lines 89-115) consisted of a convolution neural network with 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when off the center. Here is an example of off the road:

![alt text][image3]

To augment the data set, I flipped center camera image and set -steering_angle. I added left camera image and right camera image:
###### For left image, steering angle is adjusted by +0.25
###### For right image, steering angle is adjusted by -0.25
I spent very long time try to figure out the adjustment. And the final adjustment value works okay for me.

![alt text][image4]
![alt text][image5]

With flipping image, left and right image I got enough images for training.

After the collection process, I had 146,430 number of data points. I then preprocessed this data by crop the top 50 and bottom 20 of image. Because the sky and trees distort the model performance. After crop down the image size, the training is faster.

I used this training data for training the model. The model was trained with a batch size 128, 16384 samples per epoch, 20 epoch. The loss is 0.01, accuracy is about 0.24.

Here's a [link to my video result](./video.mp4)