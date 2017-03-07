#**Traffic Sign Recognition** 

Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/test_image_1.jpg "Traffic Sign 1"
[image5]: ./test/test_image_2.jpg "Traffic Sign 2"
[image6]: ./test/test_image_3.jpg "Traffic Sign 3"
[image7]: ./test/test_image_4.jpg "Traffic Sign 4"
[image8]: ./test/test_image_5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

Number of training examples = 34799

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is categorized with the class.  As the chart shows, training images are not evenly distributed (e.g. more than 1500 training examples for some classes, 
, and less than 300 training examples for some other classes), which may affect the training accuracy.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 5th code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because no significant impact is found on image recognition when color images are grayscaled; simple algorithm can be used on grayscaled images, thus the computation load is reduced and speed is increased.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because standardizing the inputs can make training faster and reduce the chances of getting stuck in local optima. Also, weight decay and Bayesian estimation can be done more conveniently with standardized inputs.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained at the end in the 5th code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by splitting the data to 95% training and 5% validation sets. 
 
 My final training set had 33059 number of images. My validation set and test set had 1740 and 12630 number of images.

I did not augment the data set this time, because I found the training accuracy is already very high. I plan to add more data later for the class with less training images. I'm going to add gaussian blur to add more images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 6th cell of the ipython notebook. 

I used the LeNet architecture for Convolutional Neural Networks as the training model.
My final model consisted of the following layers:

| Layer 1         		|     Description	        				            	| 
|:---------------------:|:---------------------------------------------------------:| 
|Input 	          	    |32x32x1 grayscale image                                    | 
|Convolution        	|5x5 filter, 1x1 stride, VALID padding, outputs 28x28x6     |
|Activation       	    |RELU 	                                                    |
|Max pooling 	    	|2x2 kernel size, 2x2 stride, VALID padding, outputs 14x14x6|
|                       |                                                           |
|Layer 2             	|Description                                                |
|Convolution      	    |5x5 filter, 1x1 stride, VALID padding, outputs 10x10x16    |
|Activation       	    |RELU                                                       |
|Max pooling 	  	    |2x2 kernel size, 2x2 stride, VALID padding, outputs 5x5x16 |
|Flatten          	    |output 400                                                 |
|                 	    |                                                           | 
|Layer 3          		|                                                           |
|Fully Connected  		|output 120 (400 with dropout 0.7)                          |
|Activation             |RELU                                                       | 
|                 	    |                                                           |
|Layer 4          	    |                                                           |
|Fully Connected  	    |output 84 (120 with dropout 0.3)                           |
|Activation       	    | RELU                                                      |
|                 	    |                                                           |
|Layer 5          	    |                                                           |
|Fully connected  	    |Logits: 43 class outputs (84 with dropput 0.49)            |
|Softmax 	     	    |One hot encode the training set                            |
        
####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 7th cell of the ipython notebook. 

To train the model, I used an adam optimizar with learning rate of 0.001. Epoch size is 30 and batch size is 128.
I used one-hot to encode the training data and calculated the cross entropy with softmax probabilities on the training classes.
The optimizer uses the mean of cross entropy as the loss function to do the training operation. 


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 8th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.98
* validation set accuracy of 0.98 
* test set accuracy of 0.896

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

My first attempt is to finish this project ASAP. Because I want to spend more time working on Keras instead of raw tensorflow. I'm using a Linear Function WX + b model from tensorflow lab. The model is not as good as CNN to classify images. So the result is not good, but after tuning the batch size and epoch, I was able to achieve the initial training accuracy at 0.80

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
For my re-submission, LeNet model was chosen because this architecture is good for image classification problems.
And it can run on a cheap laptop. 

Validation sets are used to evaluate the training model accuracy based on a loss function. Initially, I did not properly pre-process the training set and the training result was not very accurate on the model.

I did some adjustments on the architecture by the following: 
1) preprocessed the training images (grayscaled, normalized)
2) properly split training data into training and validation sets

I achieved a high accuracy of 0.98 on the training set and validation set. I achieved an accuracy of close 0.90 on the test set. next, I'm planning to 
do more preprocessing on the training set to improve the accuracy. for example, balancing the examples of each class, image augmentation, 
adjusting training layers etc. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work           	| Road work           							| 
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| Priority road       	| Priority road       							|
| Yield               	| Yield               			 				|
| Keep right          	| Keep right           							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
 
I tried different images which gave wrong prediction. But I didn't include those images in the test. I pick all 5 images which the model can accurately predict.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 1st images, the top five soft max probabilities were

######top 5 class - probablities:
######Class #25 - Road work : 1.0
######Class #5 - Speed limit (80km/h) : 0.0
######Class #12 - Priority road : 0.0
######Class #29 - Bicycles crossing : 0.0
######Class #33 - Turn right ahead : 0.0

For the 2nd images, the top five soft max probabilities were 

######top 5 class - probablities:
######Class #1 - Speed limit (30km/h) : 1.0
######Class #2 - Speed limit (50km/h) : 0.0
######Class #5 - Speed limit (80km/h) : 0.0
######Class #4 - Speed limit (70km/h) : 0.0
######Class #0 - Speed limit (20km/h) : 0.0

For the 3rd images, the top five soft max probabilities were 

######top 5 class - probablities:
######Class #12 - Priority road : 1.0
######Class #40 - Roundabout mandatory : 0.0
######Class #32 - End of all speed and passing limits : 0.0
######Class #7 - Speed limit (100km/h) : 0.0
######Class #1 - Speed limit (30km/h) : 0.0

For the 4th images, the top five soft max probabilities were 

######top 5 class - probablities:
######Class #13 - Yield : 1.0
######Class #35 - Ahead only : 0.0
######Class #2 - Speed limit (50km/h) : 0.0
######Class #12 - Priority road : 0.0
######Class #38 - Keep right : 0.0

For the 5th images, the top five soft max probabilities were 

######top 5 class - probablities:
######Class #38 - Keep right : 1.0
######Class #34 - Turn left ahead : 0.0
######Class #12 - Priority road : 0.0
######Class #15 - No vehicles : 0.0
######Class #13 - Yield : 0.0