import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def combine_thresh_debug(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # bgr = img
    # R = bgr[:, :, 2]
    # G = bgr[:, :, 1]
    # B = bgr[:, :, 0]
    # cv2.imshow('R', R)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('G', G)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('B', B)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    print(sobelx)
    abs_sobel = np.absolute(sobelx)
    print(abs_sobel)
    print(np.max(abs_sobel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    print(scaled_sobel)
    print(scaled_sobel.shape)
    plt.imshow(scaled_sobel, cmap='gray')
    plt.show()

    thresh_sobel = 20
    sob_thres = np.copy(scaled_sobel)
    sob_thres[scaled_sobel < thresh_sobel] = 0
    plt.imshow(sob_thres, cmap='gray')
    plt.show()
    # hist = cv2.calcHist([scaled_sobel], [0], None, [256], [0, 256])
    hist, bins = np.histogram(sob_thres.flatten(), 256, [thresh_sobel, 256])
    # plt.hist(scaled_sobel.ravel(), 256, [0, 256])
    plt.plot(hist)
    plt.title('Histogram for gray scale picture')
    plt.show()
    equ = cv2.equalizeHist(sob_thres)
    plt.imshow(equ, cmap='gray')
    plt.show()
    hist = cv2.calcHist([equ], [0], None, [256], [thresh_sobel, 256])
    plt.plot(hist)
    plt.title('Histogram for gray scale picture')
    plt.show()
    res = np.hstack((sob_thres, equ))
    plt.imshow(res, cmap='gray')
    plt.show()
    equ_thres = np.copy(equ)
    equ_thres[equ < 128] = 0
    plt.imshow(equ_thres, cmap='gray')
    plt.show()
    sob_combine = np.copy(equ_thres)
    sob_combine[equ_thres < 128] = 128
    sob_combine = sob_combine - 128
    plt.imshow(sob_combine, cmap='gray')
    plt.show()

    thres_s = 50
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    plt.imshow(S, cmap='gray')
    plt.show()
    H = hls[:,:,0]
    plt.imshow(H, cmap='gray')
    plt.show()
    L = hls[:,:,1]
    plt.imshow(L, cmap='gray')
    plt.show()
    hist = cv2.calcHist([S], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title('Histogram for gray scale picture')
    plt.show()
    s_thres = np.copy(S)
    s_thres[S < thres_s] = 0
    plt.imshow(s_thres, cmap='gray')
    plt.show()
    hist = cv2.calcHist([s_thres], [0], None, [256], [thres_s, 256])
    plt.plot(hist)
    plt.title('Histogram for gray scale picture')
    plt.show()
    s_combine = np.copy(s_thres)
    s_combine[s_thres < 128] = 128
    s_combine = s_combine - 128
    plt.imshow(s_combine, cmap='gray')
    plt.show()
    combined = sob_combine + s_combine
    plt.imshow(combined, cmap='gray')
    plt.show()

    com_binary = np.zeros_like(combined)
    com_binary[combined > 100] = 1
    plt.imshow(com_binary, cmap='gray')
    plt.show()

    thresh = mag_thresh
    s_binary = np.zeros_like(S)
    s_binary[(S > thresh[0]) & (S <= thresh[1])] = 1

    plt.imshow(s_binary, cmap='gray')
    plt.show()

    # Threshold x gradient
    # retval, sxbinary = cv2.threshold(scaled_sobel, 15, 250, cv2.THRESH_BINARY)
    thresh = (50, 250)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    plt.imshow(sxbinary, cmap='gray')
    plt.show()

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    # Return the binary image
    # return combined_binary
    return equ

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def combine_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # print(sobelx)
    abs_sobel = np.absolute(sobelx)
    # print(abs_sobel)
    # print(np.max(abs_sobel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # print(scaled_sobel)
    # print(scaled_sobel.shape)
    # plt.imshow(scaled_sobel, cmap='gray')
    # plt.show()

    sob_thres = np.copy(scaled_sobel)
    sob_thres[scaled_sobel < 50] = 0
    # plt.imshow(sob_thres, cmap='gray')
    # plt.show()
    # hist = cv2.calcHist([scaled_sobel], [0], None, [256], [0, 256])
    hist, bins = np.histogram(sob_thres.flatten(), 256, [50, 256])
    # plt.hist(scaled_sobel.ravel(), 256, [0, 256])
    # plt.plot(hist)
    # plt.title('Histogram for gray scale picture')
    # plt.show()
    equ = cv2.equalizeHist(sob_thres)
    # plt.imshow(equ, cmap='gray')
    # plt.show()
    hist = cv2.calcHist([equ], [0], None, [256], [50, 256])
    # plt.plot(hist)
    # plt.title('Histogram for gray scale picture')
    # plt.show()
    res = np.hstack((sob_thres, equ))
    # plt.imshow(res, cmap='gray')
    # plt.show()
    equ_thres = np.copy(equ)
    equ_thres[equ < 128] = 0
    # plt.imshow(equ_thres, cmap='gray')
    # plt.show()
    sob_combine = np.copy(equ_thres)
    sob_combine[equ_thres < 128] = 128
    sob_combine = sob_combine - 128
    # plt.imshow(sob_combine, cmap='gray')
    # plt.show()

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    # plt.imshow(S, cmap='gray')
    # plt.show()
    hist = cv2.calcHist([S], [0], None, [256], [0, 256])
    # plt.plot(hist)
    # plt.title('Histogram for gray scale picture')
    # plt.show()
    s_thres = np.copy(S)
    s_thres[S < 120] = 0
    # plt.imshow(s_thres, cmap='gray')
    # plt.show()
    hist = cv2.calcHist([s_thres], [0], None, [256], [150, 256])
    # plt.plot(hist)
    # plt.title('Histogram for gray scale picture')
    # plt.show()
    s_combine = np.copy(s_thres)
    s_combine[s_thres < 128] = 128
    s_combine = s_combine - 128
    # plt.imshow(s_combine, cmap='gray')
    # plt.show()
    combined = sob_combine + s_combine
    # plt.imshow(combined, cmap='gray')
    # plt.show()

    com_binary = np.zeros_like(combined)
    com_binary[combined > 100] = 1
    # plt.imshow(com_binary, cmap='gray')
    # plt.show()

    # Return the binary image
    return com_binary

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


# Use color transforms, gradients, etc., to create a thresholded binary image.
# Read in an image
# Choose a Sobel kernel size
ksize = 21 # Choose a larger odd number to smooth gradient measurements

img = cv2.imread('test_images/test2.jpg')
# img = cv2.undistort(img, mtx, dist, None, mtx)
combine_binary = combine_thresh(img, sobel_kernel=ksize, mag_thresh=(150, 255))
# plt.imshow(combine_binary, cmap='gray')
# plt.show()




# Apply a perspective transform to rectify binary image ("birds-eye view").
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
def unwarp(img):
    # Pass in your image into this function

    plt.imshow(img)
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

# Apply a perspective transform to rectify binary image ("birds-eye view").
img = mpimg.imread('straight_line_corner.png')
top_down_img, perspective_M, Minv = unwarp(img)
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(top_down_img)
# ax2.set_title('Undistorted and Warped Image', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# plt.show()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
myclip = VideoFileClip('project_video.mp4')
# myclip = VideoFileClip('challenge_video.mp4')
# myclip.save_frame("frame_challenge.png", t=0)
# myclip = VideoFileClip('harder_challenge_video.mp4')
for frame in myclip.iter_frames():
    # combine_binary = combine_thresh_debug(frame, sobel_kernel=ksize, mag_thresh=(150, 255))
    # # combine_binary = combine_thresh_color_sobel(frame, sobel_kernel=ksize, mag_thresh=(150, 255))
    # plt.imshow(combine_binary, cmap='gray')
    # plt.show()
    image = frame
    # image = combine_binary
    top_down_img = cv2.warpPerspective(image, perspective_M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    plt.imshow(top_down_img, cmap='gray')
    plt.show()
    combine_binary = combine_thresh_debug(top_down_img, sobel_kernel=ksize, mag_thresh=(150, 255))
    plt.imshow(combine_binary, cmap='gray')
    plt.show()
# my_clip.save_frame("frame.png", t=2)
# # white_output = 'white.mp4'
# # clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=0)
# # #clip1 = VideoFileClip("solidWhiteRight.mp4").subclip(t_start=8.32)
# # white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# # white_clip.write_videofile(white_output, audio=False)
# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip('solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)
