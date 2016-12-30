import numpy as np
import cv2

def find_lines(histogram):
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

# use soble_x and S channel strict image to calculate histogram
# extract the segment of histogram, then extract the top_down image
# poly fit to get an initial curve

# use soble_x and S channel strict image to calculate histogram
# check the histogram segments near area 600 and 700. this step is to find left/right lanes is: solid/dotted/weak/not-exist
# for each process window, calculate histogram of soble_x | S channel top_down image, get the window segment
# extract the points inside the segments. ploy fit get the initial curve.

def find_lines_pixel(img, top_down_img, top_down_img_strict):
    # print(histogram[590:630])
    height = img.shape[0]
    width = img.shape[1]
    window_num = 10
    window_size = int(height / window_num)
    histo_window = int(height / 2)
    for i in range(height - 1, histo_window, -window_size):
        img = top_down_img
        histogram = np.sum(img[i - histo_window : i, :], axis=0)
        histo_img = np.copy(img[i - histo_window: i, :])
        histogram = np.array(histogram)
        print(histogram.shape)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(histo_img, cmap='gray')
        ax1.set_title('Original Image', fontsize=50)
        ax2.plot(histogram)
        # plt.plot(histogram)
        plt.show()
        img = top_down_img_strict
        histogram_strict = np.sum(img[i - histo_window : i, :], axis=0)
        histo_img = np.copy(img[i - histo_window: i, :])
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(histo_img, cmap='gray')
        ax1.set_title('Original Image', fontsize=50)
        ax2.plot(histogram_strict)
        # plt.plot(histogram)
        plt.show()
        lines = find_lines(histogram)
        lines_strict = find_lines_strict(histogram_strict)
        lines = concat_segs(lines, histogram)
        lines_strict = concat_segs(lines_strict, histogram_strict)
        lines_merged = merge_segs(lines, lines_strict)
        start = lines_merged[0]
        end = -lines_merged[2]
        left_line_image = np.copy(img[i - window_size : i, :])
        left_line_image[:, 0:start] = 0
        left_line_image[:, end:width] = 0
        leftx = []
        for j in range(window_size):
            for k in (start, end):
                if left_line_image[j][k] == 1:
                    leftx.append([j,k])
        leftx_array = np.asarray(leftx)
        print(leftx_array.shape)
    # for i in range(height / 2 - 1, -1, -window_size):


    # return binary_output
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
lines = [539, 542, -547, 599, 605, -611, 685, 687, -701]
lines = [599, 605, -608, 688, 689, -691, 697, 697, -698, 699, 699, -700]
histogram = []

# top_down_img = mpimg.imread('top_down.png')
# top_down_img_strict = mpimg.imread('top_down_strict.png')

top_down_img = cv2.imread('top_down.png', cv2.IMREAD_GRAYSCALE)
top_down_img_strict = cv2.imread('top_down_strict.png', cv2.IMREAD_GRAYSCALE)
print(top_down_img.shape)

# img = top_down_img
# histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
# img = top_down_img_strict
# histogram_strict = np.sum(img[img.shape[0]/2:,:], axis=0)
# plt.plot(histogram)
# plt.show()
# import pickle
# with open('top_down_img.p', 'w') as f:
#     pickle.dump(top_down_img, f)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(top_down_img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down_img_strict)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

find_lines_pixel(top_down_img, top_down_img, top_down_img_strict)


lines = [539, 542, -547, 599, 605, -611, 685, 687, -701]
lines_strict = [599, 605, -608, 688, 689, -691, 697, 699, -700]
merge_segs(lines, lines_strict)

concat_segs(lines, histogram)