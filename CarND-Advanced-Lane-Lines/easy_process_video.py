import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_camera():
    # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # prepare object points
    nx = 9  # TODO: enter the number of inside corners in x
    ny = 6  # TODO: enter the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        # plt.imshow(img)
        # plt.show()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(img)
        # plt.show()

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # plt.imshow(img)
            # plt.show()
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)
            # plt.show()

    img = cv2.imread('test_images/test3.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    print(mtx)
    print(dist)
    return mtx, dist

calibration_file = 'camera_cal/calibration.p'
import os.path
print(os.path)
if os.path.isfile(calibration_file):
    print('Loading Calibration Data from pickle file...')
    with open(calibration_file, 'rb') as f:
        pickle_data = pickle.load(f)
        mtx = pickle_data['mtx']
        dist = pickle_data['dist']
        print(mtx)
        print(dist)
else:
    print('Saving data to pickle file...', calibration_file)
    try:
        with open(calibration_file, 'wb') as pfile:
            mtx, dist = calibrate_camera()
            pickle.dump(
                {
                    'mtx': mtx,
                    'dist': dist,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', calibration_file, ':', e)
        raise
    print('Calibration Data saved in pickle file.')

# Apply the distortion correction to the raw image.
images = glob.glob('test_images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    # plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    # plt.show()


# Use color transforms, gradients, etc., to create a thresholded binary image.
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def combine_thresh_color_sobel(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    print(scaled_sobel.shape)
    cv2.imshow('img', scaled_sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hls = cv2.cvtColor(scaled_sobel, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    cv2.imshow('H', H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('L', L)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('S', S)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresh = mag_thresh
    binary_output = np.zeros_like(L)
    binary_output[(L > thresh[0]) & (L <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def combine_thresh_debug(img, sobel_kernel=3, mag_thresh=(0, 255)):
    bgr = img
    R = bgr[:, :, 2]
    G = bgr[:, :, 1]
    B = bgr[:, :, 0]
    cv2.imshow('R', R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('G', G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('B', B)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    cv2.imshow('H', H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('L', L)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('S', S)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = img
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    print(scaled_sobel.shape)
    cv2.imshow('img', scaled_sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    bgr = scaled_sobel
    R = bgr[:, :, 2]
    G = bgr[:, :, 1]
    B = bgr[:, :, 0]
    cv2.imshow('H', R)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('L', G)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('S', B)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hls = cv2.cvtColor(scaled_sobel, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    cv2.imshow('H', H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('L', L)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('S', S)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    thresh = (105, 255)
    binary_output = np.zeros_like(L)
    binary_output[(L > thresh[0]) & (L <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    # Here I'm suppressing annoying error messages
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def combine_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    print(scaled_sobel.shape)
    # plt.imshow(scaled_sobel, cmap='gray')
    # plt.show()
    thresh = (50, 250)
    G = img[:, :, 1]
    g_binary = np.zeros_like(G)
    g_binary[(G > thresh[0]) & (G <= thresh[1])] = 1
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    # plt.imshow(S, cmap='gray')
    # plt.show()
    thresh = mag_thresh
    s_binary = np.zeros_like(S)
    s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    # plt.imshow(s_binary, cmap='gray')
    # plt.show()

    # Threshold x gradient
    # retval, sxbinary = cv2.threshold(scaled_sobel, 15, 250, cv2.THRESH_BINARY)
    thresh = (50, 250)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # plt.imshow(sxbinary, cmap='gray')
    # plt.show()

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) | (sxbinary == 1)) & (g_binary == 1)] = 1
    # Return the binary image
    return combined_binary

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def combine_thresh_strict(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    print(scaled_sobel.shape)
    # plt.imshow(scaled_sobel, cmap='gray')
    # plt.show()

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    # plt.imshow(S, cmap='gray')
    # plt.show()
    thresh = mag_thresh
    s_binary = np.zeros_like(S)
    s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    # plt.imshow(s_binary, cmap='gray')
    # plt.show()

    # Threshold x gradient
    # retval, sxbinary = cv2.threshold(scaled_sobel, 15, 250, cv2.THRESH_BINARY)
    thresh = (50, 250)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # plt.imshow(sxbinary, cmap='gray')
    # plt.show()

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) & (sxbinary == 1)] = 1
    # Return the binary image
    return combined_binary

# Apply a perspective transform to rectify binary image ("birds-eye view").
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
def unwarp(img):
    # Pass in your image into this function

    # plt.imshow(img)
        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
             #Note: you could pick any four of the detected corners
             # as long as those four corners define a rectangle
             #One especially smart way to do this would be to use four well-chosen
             # corners that were automatically detected during the undistortion steps
             #We recommend using the automatic detection of corners in your code
    src = np.float32([[631,429], [650,429], [1126,690], [286,690]])
    src = np.float32([[590, 461], [708, 461], [1126, 690], [286, 690]])
    src = np.float32([[350, 640], [1033, 640], [1126, 690], [286, 690]])
    src = np.float32([[480, 540], [855, 540], [1126, 690], [286, 690]])
    src = np.float32([[590, 460], [715, 460], [1126, 690], [286, 690]])
    src = np.float32([[641, 425], [643, 425], [1126, 690], [286, 690]])
    src = np.float32([[590, 461], [708, 461], [1126, 690], [286, 690]])
        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    print(img.shape)
    width = img.shape[1]
    height = img.shape[0]
    dst = np.float32([[286,0], [1126,0], [1126,height], [286,height]])
    dst = np.float32([[286, 500], [1126, 500], [1126, height], [286, height]])
    dst = np.float32([[0, 0], [760, 0], [760, 100], [0, 100]])
    dst = np.float32([[0, 0], [740, 0], [740, 150], [0, 150]])
    dst = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])
    dst = np.float32([[600, 550], [700, 550], [700, height], [600, height]])
    dst = np.float32([[600, 0], [700, 0], [700, height], [600, height]])
        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
        # e) use cv2.warpPerspective() to warp your image to a top-down view
    img_size = (width, height)
    # warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(img, M, (760,100))
    warped = cv2.warpPerspective(img, M, (740, 150))
    warped = cv2.warpPerspective(img, M, (300, 400))
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def find_lines(histogram):
    in_segment = False
    seg_max = 0
    lines = []
    # for i in range(len(histogram)):
    for i in range(580,750):
        if in_segment is True:
            if (histogram[i] <= 0):
                in_segment = False
                lines.append(seg_index)
                lines.append(-i)
            else:
                if (histogram[i] > seg_max):
                    seg_max = histogram[i]
                    seg_index = i
                # seg_max = max(seg_max, histogram[i])

        elif (histogram[i] > 20):
            in_segment = True
            seg_max = histogram[i]
            seg_index = i
            lines.append(i)
            # new
    if (len(lines) % 3 != 0):
        lines.append(seg_max)
        lines.append(len(histogram))
    print(lines)
    return lines

def concat_segs(lines, histogram):
    concat_lines = []
    prev_seg = lines[0:3]
    for i in range(5, len(lines), 3):
        start = lines[i-2]
        end = lines[i]
        if start >= (-prev_seg[2] + 3):
            concat_lines.append(prev_seg[0])
            concat_lines.append(prev_seg[1])
            concat_lines.append(prev_seg[2])
            prev_seg = lines[i-2:i+1]
        else:
            prev_seg[2] = end
            if (histogram[lines[i-1]] > histogram[prev_seg[1]]):
                prev_seg[1] = lines[i-1]
    concat_lines.append(prev_seg[0])
    concat_lines.append(prev_seg[1])
    concat_lines.append(prev_seg[2])
    print(concat_lines)
    return concat_lines

def find_lines_strict(histogram):
    in_segment = False
    seg_max = 0
    lines = []
    for i in range(len(histogram)):
        if in_segment is True:
            if (histogram[i] <= 0):
                in_segment = False
                lines.append(seg_index)
                lines.append(-i)
            else:
                if (histogram[i] > seg_max):
                    seg_max = histogram[i]
                    seg_index = i
                # seg_max = max(seg_max, histogram[i])

        elif (histogram[i] > 0):
            in_segment = True
            seg_max = histogram[i]
            seg_index = i
            lines.append(i)
            # new
    if (len(lines) % 3 != 0):
        lines.append(seg_max)
        lines.append(len(histogram))
    print(lines)
    return lines

def merge_segs(lines, lines_strict):
    merged_lines = []
    index_a = 0
    index_b = 0
    len_a = len(lines)
    len_b = len(lines_strict)
    prev_seg = []
    while index_a < len_a and index_b < len_b:
        if index_a < len_a:
            start_a = lines[index_a]
            end_a = lines[index_a + 2]
        if index_b < len_b:
            start_b = lines_strict[index_b]
            end_b = lines_strict[index_b + 2]
        if start_a <= start_b and -end_a >= -end_b:
            merged_lines.append(start_a)
            merged_lines.append(lines[index_a + 1])
            merged_lines.append(end_a)
            index_b = index_b + 3
        index_a = index_a + 3

    print(merged_lines)
    return merged_lines


def find_lines_pixel_fixme(img, top_down_img):
    # plt.imshow(top_down_img, cmap='gray')
    # plt.show()
    # print(histogram[590:630])
    height = img.shape[0]
    width = img.shape[1]
    window_num = 10
    window_size = int(height / window_num)
    histo_window = int(height / 2)
    left = []
    right = []
    for i in range(height - 1, histo_window, -window_size):
        img = top_down_img
        histogram = np.sum(img[i - histo_window : i, :], axis=0)
        histo_img = np.copy(img[i - histo_window: i, :])
        histogram = np.array(histogram)
        # print(histogram.shape)
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(histo_img, cmap='gray')
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.plot(histogram)
        # # plt.plot(histogram)
        # plt.show()
        # # img = top_down_img_strict
        # # histogram_strict = np.sum(img[i - histo_window : i, :], axis=0)
        # # histo_img = np.copy(img[i - histo_window: i, :])
        # # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # # f.tight_layout()
        # # ax1.imshow(histo_img, cmap='gray')
        # # ax1.set_title('Original Image', fontsize=50)
        # # ax2.plot(histogram_strict)
        # # # plt.plot(histogram)
        # # plt.show()
        lines = find_lines(histogram)
        # lines_strict = find_lines_strict(histogram_strict)
        # lines = concat_segs(lines, histogram)
        # lines_strict = concat_segs(lines_strict, histogram_strict)
        # lines_merged = merge_segs(lines, lines_strict)
        if len(lines) >= 3:
            start = lines[0]
            end = -lines[2]
            left_line_image = np.copy(img[i - window_size : i, :])
            right_line_image = np.copy(img[i - window_size: i, :])
            left_line_image[:, 0:start] = 0
            left_line_image[:, end:width] = 0

            left_line_image_top = np.copy(img[i + 1 - histo_window : i + 1 - histo_window + window_size, :])
            right_line_image_top = np.copy(img[i + 1 - histo_window : i + 1 - histo_window + window_size, :])
            left_line_image_top[:, 0:start] = 0
            left_line_image_top[:, end:width] = 0

            for j in range(window_size):
                for k in range(start, end):
                    if left_line_image[j][k] == 1:
                        left.append([j + i - window_size,k])
                    if left_line_image_top[j][k] == 1:
                        left.append([j + i - window_size - histo_window, k])

        if len(lines) >= 6:
            startr = lines[3]
            endr = -lines[5]

            right_line_image[:, 0:startr] = 0
            right_line_image[:, endr:width] = 0

            right_line_image_top[:, 0:startr] = 0
            right_line_image_top[:, endr:width] = 0
            for j in range(window_size):
                for k in range(startr, endr):
                    if right_line_image[j][k] == 1:
                        right.append([j + i - window_size,k])
                    if right_line_image_top[j][k] == 1:
                        right.append([j + i - window_size - histo_window,k])
        # plt.imshow(right_line_image_top, cmap='gray')
        # plt.show()
        # plt.imshow(left_line_image_top, cmap='gray')
        # plt.show()

    left_array = np.asarray(left)
    right_array = np.asarray(right)
        # print(leftx_array.shape)    # for i in range(height / 2 - 1, -1, -window_size):

    return left_array, right_array
    # return binary_output


def find_lines_pixel(img, top_down_img):
    # plt.imshow(top_down_img, cmap='gray')
    # plt.show()
    # print(histogram[590:630])
    height = img.shape[0]
    width = img.shape[1]
    window_num = 10
    window_size = int(height / window_num)
    histo_window = int(height / 2)
    left = []
    right = []
    for i in range(height - 1, histo_window, -window_size):
        img = top_down_img
        histogram = np.sum(img[i - histo_window : i, :], axis=0)
        histo_img = np.copy(img[i - histo_window: i, :])
        histogram = np.array(histogram)
        # print(histogram.shape)
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(histo_img, cmap='gray')
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.plot(histogram)
        # # plt.plot(histogram)
        # plt.show()
        # # img = top_down_img_strict
        # # histogram_strict = np.sum(img[i - histo_window : i, :], axis=0)
        # # histo_img = np.copy(img[i - histo_window: i, :])
        # # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # # f.tight_layout()
        # # ax1.imshow(histo_img, cmap='gray')
        # # ax1.set_title('Original Image', fontsize=50)
        # # ax2.plot(histogram_strict)
        # # # plt.plot(histogram)
        # # plt.show()
        # lines = find_lines(histogram)
        # lines_strict = find_lines_strict(histogram_strict)
        # lines = concat_segs(lines, histogram)
        # lines_strict = concat_segs(lines_strict, histogram_strict)
        # lines_merged = merge_segs(lines, lines_strict)
        # if len(lines) >= 3:
        start = 580
        end = 620
        left_line_image = np.copy(img[i - window_size : i, :])
        right_line_image = np.copy(img[i - window_size: i, :])
        left_line_image[:, 0:start] = 0
        left_line_image[:, end:width] = 0

        left_line_image_top = np.copy(img[i + 1 - histo_window : i + 1 - histo_window + window_size, :])
        right_line_image_top = np.copy(img[i + 1 - histo_window : i + 1 - histo_window + window_size, :])
        left_line_image_top[:, 0:start] = 0
        left_line_image_top[:, end:width] = 0

        for j in range(window_size):
            for k in range(start, end):
                if left_line_image[j][k] == 1:
                    left.append([j + i - window_size,k])
                if left_line_image_top[j][k] == 1:
                    left.append([j + i - window_size - histo_window, k])

        # if len(lines) >= 6:
        startr = 680
        endr = 720

        right_line_image[:, 0:startr] = 0
        right_line_image[:, endr:width] = 0

        right_line_image_top[:, 0:startr] = 0
        right_line_image_top[:, endr:width] = 0
        for j in range(window_size):
            for k in range(startr, endr):
                if right_line_image[j][k] == 1:
                    right.append([j + i - window_size,k])
                if right_line_image_top[j][k] == 1:
                    right.append([j + i - window_size - histo_window,k])
        # plt.imshow(right_line_image_top, cmap='gray')
        # plt.show()
        # plt.imshow(left_line_image_top, cmap='gray')
        # plt.show()

    left_array = np.asarray(left)
    right_array = np.asarray(right)
        # print(leftx_array.shape)    # for i in range(height / 2 - 1, -1, -window_size):

    return left_array, right_array
    # return binary_output

# Early in the code before pipeline
polygon_points_old = None
warp_old = None

def process_image(frame):
    global polygon_points_old
    global warp_old
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # Read in an image
    # Choose a Sobel kernel size
    ksize = 9  # Choose a larger odd number to smooth gradient measurements

    img = frame
    img = cv2.undistort(img, mtx, dist, None, mtx)
    undist = img
    combine_binary = combine_thresh(img, sobel_kernel=ksize, mag_thresh=(150, 255))
    combine_binary_strict = combine_thresh_strict(img, sobel_kernel=ksize, mag_thresh=(150, 255))
    # plt.imshow(combine_binary, cmap='gray')
    # plt.show()

    # image = mpimg.imread('test_images/test3.jpg')
    # image = mpimg.imread('test_images/test2.jpg')
    # image = mpimg.imread('frame_challenge.png')
    # image = mpimg.imread('frame.png')
    # if False:
    #     # Apply each of the thresholding functions
    #     gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(25, 255))
    #     grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25, 255))
    #     mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(25, 255))
    #     dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.2))
    #
    #     # Plot the result
    #     # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    #
    #     f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(24, 9))
    #     f.tight_layout()
    #     ax1.imshow(image)
    #     ax1.set_title('Original Image', fontsize=50)
    #     ax2.imshow(gradx, cmap='gray')
    #     ax2.set_title('gradx', fontsize=15)
    #     ax3.imshow(grady, cmap='gray')
    #     ax3.set_title('grady', fontsize=15)
    #     ax4.imshow(mag_binary, cmap='gray')
    #     ax4.set_title('Magnitude', fontsize=15)
    #     ax5.imshow(dir_binary, cmap='gray')
    #     ax5.set_title('directional', fontsize=15)
    #     ax6.imshow(mag_binary, cmap='gray')
    #     ax6.set_title('Thresholded Gradient', fontsize=15)
    #     # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #     plt.show()
    #
    #     hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #     H = hls[:, :, 0]
    #     L = hls[:, :, 1]
    #     S = hls[:, :, 2]
    #
    #     thresh = (105, 255)
    #     binary_S = np.zeros_like(S)
    #     binary_S[(S > thresh[0]) & (S <= thresh[1])] = 1
    #
    #     thresh = (1, 90)
    #     binary_H = np.zeros_like(S)
    #     binary_H[(H > thresh[0]) & (H <= thresh[1])] = 1
    #     # Lzero = np.zeros_like(L)
    #     # hs = np.vstack(H, Lzero, S)
    #     f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(24, 9))
    #     f.tight_layout()
    #     ax1.imshow(image)
    #     ax1.set_title('Original Image', fontsize=50)
    #     ax2.imshow(H, cmap='gray')
    #     ax2.set_title('H', fontsize=15)
    #     ax3.imshow(L, cmap='gray')
    #     ax3.set_title('L', fontsize=15)
    #     ax4.imshow(S, cmap='gray')
    #     ax4.set_title('S', fontsize=15)
    #     ax5.imshow(binary_S, cmap='gray')
    #     ax5.set_title('binary_S', fontsize=15)
    #     ax6.imshow(binary_H, cmap='gray')
    #     ax6.set_title('binary_H', fontsize=15)
    #     # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #     plt.show()

    # ax1 = plt.subplot(511)
    # ax1.imshow(image)
    # ax1.set_title('Original Image', fontsize=50)
    #
    # ax2 = plt.subplot(512, figsize=(24, 9))
    # ax2.imshow(mag_binary, cmap='gray')
    # ax2.set_title('Thresholded Gradient', fontsize=50)
    #
    # # plt.tight_layout()
    # plt.show()


    # from moviepy.editor import VideoFileClip
    # clip = VideoFileClip('project_video.mp4')
    # clip.save_frame("frame.png", t=15)
    if False:
        img = cv2.imread("frame.png")
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.show()
        cv2.imwrite('straight_line.png', dst)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    img = mpimg.imread('straight_line_corner.png')
    top_down_img, perspective_M, Minv = unwarp(img)
    if False:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(top_down_img)
        ax2.set_title('Undistorted and Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    # top_down_img = cv2.warpPerspective(mag_binary, perspective_M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    top_down_img = cv2.warpPerspective(combine_binary, perspective_M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    # top_down_img_strict = cv2.warpPerspective(combine_binary_strict, perspective_M, (img.shape[1], img.shape[0]),
    #                                           flags=cv2.INTER_LINEAR)
    # # warped = cv2.cvtColor(top_down_img, cv2.COLOR_RGB2GRAY)
    # # Plot the result
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(image)
    # ax1.set_title('Original Image', fontsize=50)
    # ax2.imshow(top_down_img, cmap='gray')
    # ax2.set_title('Thresholded Gradient', fontsize=50)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # plt.show()
    #
    # # plt.imsave('top_down.png', top_down_img, cmap = plt.cm.gray)
    # # plt.imsave('top_down_strict.png', top_down_img_strict, cmap = plt.cm.gray)
    #
    # cv2.imwrite('top_down.png', top_down_img)
    # cv2.imwrite('top_down_strict.png', top_down_img_strict)

    img = top_down_img
    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
    # img = top_down_img_strict
    # histogram_strict = np.sum(img[img.shape[0] / 2:, :], axis=0)
    # plt.plot(histogram)
    # plt.show()

    lines = find_lines(histogram)
    # lines_strict = find_lines_strict(histogram_strict)
    # lines = concat_segs(lines, histogram)
    # lines_strict = concat_segs(lines_strict, histogram_strict)
    # lines_merged = merge_segs(lines, lines_strict)




    left, right = find_lines_pixel(top_down_img, top_down_img)



    # # Generate some fake data to represent lane-line pixels
    # yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
    # leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
    #                               for idx, elem in enumerate(yvals)])
    # leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    # rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51)
    #                                 for idx, elem in enumerate(yvals)])
    # rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    if len(left) < 10 or len(right) < 10:
        result = cv2.addWeighted(undist, 1, warp_old, 0.3, 0)
        return result

    lefty = left[:, 0]
    leftx = left[:, 1]

    righty = right[:, 0]
    rightx = right[:, 1]

    fity = np.arange(img.shape[0])

    # plt.plot(leftx, lefty, color='green', linewidth=3)
    # plt.plot(rightx, righty, color='green', linewidth=3)
    # plt.show()

    # Fit a second order polynomial to each fake lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

    # plt.plot(leftx, lefty, 'o', color='red')
    # plt.plot(rightx, righty, 'o', color='blue')
    # plt.xlim(0, 1280)
    # plt.ylim(0, 720)
    # plt.plot(left_fitx, fity, color='green', linewidth=3)
    # plt.plot(right_fitx, fity, color='green', linewidth=3)
    # plt.gca().invert_yaxis()  # to visualize as we do the images
    # plt.show()
    #
    # # Define y-value where we want radius of curvature
    # # I'll choose the maximum y-value, corresponding to the bottom of the image
    # y_eval = np.max(lefty)
    # left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) \
    #                 / np.absolute(2 * left_fit[0])
    # y_eval = np.max(righty)
    # right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) \
    #                  / np.absolute(2 * right_fit[0])
    # print(left_curverad, right_curverad)
    # # Example values: 1163.9    1213.7
    #
    #
    # #
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    y_eval = np.max(lefty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    y_eval = np.max(lefty)
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 3380.7 m    3189.3 m

    bottom_y = img.shape[0]
    center = img.shape[1] / 2
    left_lane_pix = left_fit[0] * bottom_y ** 2 + left_fit[1] * bottom_y + left_fit[2]
    right_lane_pix = right_fit[0] * bottom_y ** 2 + right_fit[1] * bottom_y + right_fit[2]
    lane_center_pix = (left_lane_pix + right_lane_pix) / 2
    lane_ceter_meter = xm_per_pix * (lane_center_pix - center)
    print(lane_ceter_meter)


    # Create an image to draw the lines on
    # undist = image
    warped = top_down_img
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
    # pts_right = np.array([np.transpose(np.vstack([right_fitx, fity]))])
    pts = np.hstack((pts_left, pts_right))

    center_x = np.empty(img.shape[0])
    center_x.fill(img.shape[1] / 2)
    pts_center = np.array([np.transpose(np.vstack([center_x, fity]))])
    right_pts = np.hstack((pts_center, pts_right))
    right_warp = np.zeros_like(warped).astype(np.uint8)
    cv2.fillPoly(right_warp, np.int_([right_pts]), 255)
    # plt.imshow(right_warp)
    # plt.show()

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # plt.imshow(color_warp)
    # plt.show()
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))


    # # Convert to grayscale
    # polygon_points = cv2.cvtColor(color_warp, cv2.COLOR_RGB2GRAY)
    #
    # # In the pipeline
    #
    # if (polygon_points_old == None):
    #     polygon_points_old = polygon_points
    #
    # a = polygon_points_old
    # b = polygon_points
    # ret = cv2.matchShapes(a, b, 1, 0.0)
    # print('ret ', ret)
    # if (ret < 0.0035):
    #     # Use the new polygon points to write the next frame due to similarites of last sucessfully written polygon area
    #
    #     polygon_points_old = polygon_points
    #     warp_old = newwarp
    #
    # else:
    # # Use the old polygon points to write the next frame due to iregulatires
    # # Then write the out the old poly gon points
    # # This will help only use your good detections
    #     newwarp = warp_old

    # Combine the result with the original image
    # plt.imshow(newwarp)
    # plt.show()
    # plt.imshow(undist)
    # plt.show()
    # newwarp = newwarp.astype(np.float32)
    print(newwarp.shape)
    print(undist.shape)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    # plt.show()

    curverad = (left_curverad + right_curverad) / 2
    curverad_str = "{} {:.0f} {}".format("Radius of Curvature = ", curverad, "(m)")
    center_str = "{} {:.2f} {}".format("Vehicle is ", lane_ceter_meter, "m left of center")
    # "Radius of Curvature = " curverad "(m)"
    # "Vehicle is " lane_ceter_meter "m left of center"
    cv2.putText(result,curverad_str, (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    cv2.putText(result, center_str, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    # plt.imshow(result)
    # plt.show()
    return result



# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
myclip = VideoFileClip('project_video.mp4')
# myclip = VideoFileClip('project_video.mp4').subclip(t_start=23.00)
# myclip = VideoFileClip('project_video.mp4').subclip(t_start=40.59)
# myclip = VideoFileClip('challenge_video.mp4')
# myclip = VideoFileClip('harder_challenge_video.mp4')
# for frame in myclip.iter_frames():
#     process_image(frame)

white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=0)
#clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=8.32)
white_clip = myclip.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

''''
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None






def process_image(image):
    initial_image = np.copy(image)
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    # plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    # plt.show()




def process_image_file(img_file):
    # reading in an image
    image = mpimg.imread(img_file)
    return process_image(image)

import os
# image_files = os.listdir("test_images/")
# for file in image_files:
#     ret_image = process_image_file("test_images/" + file)
#     cv2.imwrite("marked_" + file, ret_image)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
clip = VideoFileClip('project_video.mp4')
my_clip.save_frame("frame.png", t=2)
# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=0)
# #clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=8.32)
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)
'''