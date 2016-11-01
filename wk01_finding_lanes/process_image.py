#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., lamda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + lamda
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)


def fitting_line(gray, region_topy = 325):
    # Find Contours and fitline to average lines
    im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        cnt = np.vstack(con)

    # then apply fitline() function
    #return cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # Now find two extreme points on the line to draw line
    topx = int(((region_topy-y) * vx / vy) + x)
    bottomx = int(((gray.shape[0] - y) * vx / vy) + x)

    #print(left_masked_edges.shape[1])
    # Finally draw the line
    img = np.zeros_like(gray)
    cv2.line(img, (bottomx, gray.shape[0] - 1), (topx, region_topy), 255, 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def process_image(image):
    initial_image = np.copy(image)
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimesions:', image.shape)
    plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    plt.show()

    image = grayscale(image)
    plt.imshow(image, cmap='gray')
    plt.show()
    image = gaussian_blur(image, 7)
    image = canny(image, 50, 150)
    plt.imshow(image, cmap='gray')  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    plt.show()

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(470, 290), (480, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(image, vertices)
    plt.imshow(masked_edges, cmap='gray')
    plt.show()
    #masked_edges = cv2.bitwise_and(edges, mask)
    line_image = hough_lines(masked_edges, 1, np.pi/180, 30, 20, 20)
    plt.imshow(line_image, cmap='gray')
    plt.show()

    mask = np.zeros_like(line_image)
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((line_image, mask, mask))
    marked_image = weighted_img(color_edges, initial_image)
    plt.imshow(marked_image)
    plt.show()

    # Seperate left lane and right lane lines
    left_vertices = np.array([[(0,imshape[0]),(470, 310), (470, imshape[0])]], dtype=np.int32)
    right_vertices = np.array([[(490, 310), (490, imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    left_masked_edges = region_of_interest(line_image, left_vertices)
    right_masked_edges = region_of_interest(line_image, right_vertices)
    plt.imshow(left_masked_edges, cmap='gray')
    plt.show()
    plt.imshow(right_masked_edges, cmap='gray')
    plt.show()

    left_line_img = fitting_line(left_masked_edges)
    # Create a "color" binary image to combine with line image
    color_fit_line = np.dstack((left_line_img, mask, mask))
    left_marked_image = weighted_img(color_fit_line, initial_image)
    plt.imshow(left_marked_image)
    plt.show()

    right_line_img = fitting_line(right_masked_edges)
    # Create a "color" binary image to combine with line image
    right_color_fit_line = np.dstack((right_line_img, mask, mask))
    right_marked_image = weighted_img(right_color_fit_line, left_marked_image)
    plt.imshow(right_marked_image)
    plt.show()
    return right_marked_image

def process_image_file(img_file):
    # reading in an image
    image = mpimg.imread(img_file)
    return process_image(image)

import os
image_files = os.listdir("test_images/")
for file in image_files:
    process_image_file("test_images/" + file)
