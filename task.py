import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

lprev = [0, 0]
rprev = [0, 0]
lcoefmean = 1
rcoefmean = 1
def extention_lines(img, rho, theta, threshold, min_line_len, max_line_gap, region):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    left_line = [[], []]
    right_line = [[],[]]

    global lprev, rprev, lcoefmean, rcoefmean

    for line in lines:
        coef = (line[0][3] - line[0][1]) / (line[0][2] / line[0][0])
        if coef < 0:
            lcoefmean += coef
            lcoefmean /= 2
            if (coef / lcoefmean) > 0.2: 
                left_line[1].extend(line[0][1::2])
                left_line[0].extend(line[0][::2])
        else:
            rcoefmean += coef
            rcoefmean /= 2
            if (coef / rcoefmean) > 0.2: 
                right_line[1].extend(line[0][1::2])
                right_line[0].extend(line[0][::2])

    l = np.polyfit(left_line[0], left_line[1], 1)
    l = np.mean(np.array([lprev, l]), axis=0)
    lprev = l
    l = np.poly1d(l)

    r = np.polyfit(right_line[0], right_line[1], 1)
    r = np.mean(np.array([rprev, r]), axis=0)
    rprev = r
    r = np.poly1d(r)   

    lines = []
    lines.append([region[0][0][0], int(l(region[0][0][0])), region[0][3][0], int(l(region[0][3][0]))])
    lines.append([region[0][1][0], int(r(region[0][1][0])), region[0][2][0], int(r(region[0][2][0]))])

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, [lines])
    return line_img

def weighted_img(img, initial_img, a=0.8, b=1., g=0.):
    return cv2.addWeighted(initial_img, a, img, b, g)

def process_image(image):
    result = grayscale(image)
    result = gaussian_blur(result, 7)
    result = canny(result, 100, 200)
    region = np.array( [[[415,330],[545,330],[900,535],[120,535]]], dtype=np.int32 )
    result = region_of_interest(result, region)
    result = extention_lines(result, 1, 0.03, 10, 20, 3, region)
    result = weighted_img(result, image)    
    return result

# white_output = 'test_videos_output/solidWhiteRight.mp4'
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_output = 'test_videos_output/solidYellowLeft.mp4'
clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
# white_output = 'test_videos_output/challenge.mp4'
# clip1 = VideoFileClip("test_videos/challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)