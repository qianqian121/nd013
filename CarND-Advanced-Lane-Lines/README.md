## Advanced Lane Finding

The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  The video called `project_video.mp4` is the video your pipeline should work well on.  `challenge_video.mp4` is an extra (and optional) challenge for you if you want to test your pipeline.

If you're feeling ambitious (totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!



[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---
###Writeup / README

###Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in lines #8 through #58 of the file called `easy_process_video.py`.  

###Pipeline (single images)

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #267 through #304 in `easy_process_video.py`).  

The code for my perspective transform includes a function called `warper()`, which appears in lines #348 through #387 in the file `easy_process_video.py`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

####1. Step of the result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `easy_process_video.py` line #889 to line #909.

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./white.mp4)

---

###Discussion

#### I spent lots of effort worked out a very robust warper() function. It works really well for small curves. For sharp curves, multiple segmentation warper() function to be implemented.
#### For lane detection. To switch Sobel operator with Schar operator. To use segmentation to remove dark pitch on the road for challenge.mp4. To apply warper() funtion then apply edge detection. To implement OpenCV curve lane hough() to detect lane lines more robustly.

