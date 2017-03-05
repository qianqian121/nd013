import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
    
# # Read in cars and notcars
# # images = glob.glob('*.jpeg')
# import glob2
# images = glob2.glob("training_pics/**/*.jpeg")
# cars = []
# notcars = []
# for image in images:
#     if 'image' in image or 'extra' in image:
#         notcars.append(image)
#     else:
#         cars.append(image)

# Read in cars and notcars
cars = glob.glob('training_pics/vehicles/**/*.png')
notcars = glob.glob('training_pics/non-vehicles/**/*.png')

# Check that arrays are not empty
print(cars[0])
print(notcars[0])


# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
# sample_size = 500
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

# Spatial Binning of Color: size = (16, 16)
# Histograms of Color: nbins = 32
# Histogram of Oriented Gradient (HOG): orient = 8, pix_per_cell = 8, cell_per_block = 2

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()

def create_windows(pyramid, image_size):
    output = []
    for w_size, y_lims in pyramid:
        windows = slide_window(image_size, x_start_stop=[700, 1280], y_start_stop=y_lims,
                        xy_window=w_size, xy_overlap=(0.5, 0.5))
        output.extend(windows)
    return output

pyramid = [
            # ((64, 64),  [400, 500]),
            ((96, 64),  [400, 500]),
           # ((96, 96),  [400, 500]),
           ((128, 64), [400, 500]),
           # ((128, 128),[450, 578]),
           ((192, 128),[450, 650]),
#             ((256, 256),[450, None])
      ]

# pyramid = [
#            ((96, 96),  [400, 500])
#            # ((128, 128),[450, 578]),
#            # ((192, 192),[450, None]),
# #             ((256, 256),[450, None])
#       ]
image_size = (720, 1280)
windows = create_windows(pyramid, image_size)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    xstart = 656
    draw_img = np.copy(img[:, xstart:1280, :])
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, xstart:1280, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Initialize a list to append window positions to
    window_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                # cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                #               (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                # Append window position to list
                window_list.append(((xstart+xbox_left, ytop_draw + ystart), (xstart+xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    # return draw_img
    return window_list

ystart = 400
ystop = 656
scale = 1.5


def process_image_draw(image):
    out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    return out_img

history_window_one = None
history_window_two = None

def get_labeled_bboxes(labels):
    # Iterate through all detected cars
    boxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        # cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        boxes.append(bbox)
    # Return the image
    return boxes

def add_heat_val(heatmap, bbox_list, val):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += val

    # Return updated heatmap
    return heatmap

def process_image(image):
    global history_window_one
    global history_window_two
    hot_windows = []
    hot_windows1p5 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)

    hot_windows.extend(hot_windows1p5)

    hot_windows1p25 = find_cars(image, ystart, ystop, 1.25, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    hot_windows.extend(hot_windows1p25)

    hot_windows1p1 = find_cars(image, ystart, ystop, 1.1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)
    hot_windows.extend(hot_windows1p1)

    draw_image = np.copy(image)
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # plt.imshow(window_img)
    # plt.show()

    from scipy.ndimage.measurements import label
    # heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = np.zeros_like(image[:,:,0])
    heatmap = add_heat(heat, hot_windows)
    heatmap = apply_threshold(heatmap, 5)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    # plt.imshow(labels[0], cmap='gray')
    # plt.show()
    # Draw bounding boxes on a copy of the image
    # draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # Display the image
    # plt.imshow(draw_img)
    # plt.show()
    current_heat_windows = get_labeled_bboxes(labels)

    heat = np.zeros_like(image[:, :, 0])
    if history_window_one is not None:
        heatmap = add_heat_val(heat, history_window_one, val=10)
    if history_window_two is not None:
        heatmap = add_heat_val(heatmap, history_window_two, val=10)
    heatmap = add_heat_val(heatmap, current_heat_windows, val=1)
    heatmap = apply_threshold(heatmap, 9)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    # plt.imshow(labels[0], cmap='gray')
    # plt.show()

    history_window_one = history_window_two
    history_window_two = current_heat_windows

    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # plt.imshow(draw_img)
    # plt.show()
    return draw_img


def process_image_raw(image):
    # global windows
    # image = mpimg.imread('bbox-example-image.jpg')
    draw_image = np.copy(image)

    out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                        hist_bins)

    # plt.imshow(out_img)
    # plt.show()
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255

    # windows = slide_window(image, x_start_stop=[700, 1280], y_start_stop=y_start_stop,
    #                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
    #
    # plt.imshow(window_img)
    # plt.show()


    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # plt.imshow(window_img)
    # plt.show()

    from scipy.ndimage.measurements import label
    # heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = np.zeros_like(image[:,:,0])
    heatmap = add_heat(heat, hot_windows)
    heatmap = apply_threshold(heatmap, 1)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    plt.imshow(labels[0], cmap='gray')
    # plt.show()
    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # # Display the image
    # plt.imshow(draw_img)
    # plt.show()
    return draw_img


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
myclip = VideoFileClip('project_video.mp4')
# myclip = VideoFileClip('project_video.mp4').subclip(t_start=46.00)
# myclip = VideoFileClip('project_video.mp4').subclip(t_start=23.00)
# myclip = VideoFileClip('project_video.mp4').subclip(t_start=23.00 + 18.80)
# myclip = VideoFileClip('challenge_video.mp4')
# myclip = VideoFileClip('harder_challenge_video.mp4')
# for frame in myclip.iter_frames():
#     process_image(frame)

white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=0)
#clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=8.32)
white_clip = myclip.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)