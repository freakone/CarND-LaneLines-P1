import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

MEAN_WINDOW = 4

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
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


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
     return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

def mean_window(arr, item):
    arr.append(item)
    if len(arr) > MEAN_WINDOW:
        arr.pop(0)
    return arr

def coef_filtering(lines):
    proper_lines = []
    for line in lines:
        coef = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
        if abs(coef) > 0.4:
            proper_lines.append(line)
    return proper_lines

def get_equation(line, prev):
    l = np.polyfit(line[0], line[1], 1)
    l = np.mean(np.array(mean_window(prev, l)), axis=0)
    return np.poly1d(l)

lprev = []
rprev = []
def extention_lines(img, lines, region):
    try: 
        lines = coef_filtering(lines)        
        left_line = [[], []]
        right_line = [[],[]]
        global lprev, rprev
        for line in lines:
            coef = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
            if coef < 0:
                left_line[1].extend(line[0][1::2])
                left_line[0].extend(line[0][::2])
            else:
                right_line[1].extend(line[0][1::2])
                right_line[0].extend(line[0][::2])

        l = get_equation(left_line, lprev)
        r = get_equation(right_line, rprev)

        lines = []
        lines.append([region[0][0][0], int(l(region[0][0][0])), region[0][3][0], int(l(region[0][3][0]))])
        lines.append([region[0][1][0], int(r(region[0][1][0])), region[0][2][0], int(r(region[0][2][0]))])

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, [lines])      
    except TypeError:
        print("Segmentation error!")
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    return line_img
   

def weighted_img(img, initial_img, a=0.8, b=1., g=0.):
    return cv2.addWeighted(initial_img, a, img, b, g)

def process_image(image):
    result = grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    result = clahe.apply(result)
    result = gaussian_blur(result, 7)
    result = canny(result, 70, 170)

    x = image.shape[1]
    y = image.shape[0]
    region = np.array( [[[0.48*x,0.55*y],[0.52*x,0.55*y],[0.9*x,y],[0.10*x,y]]], dtype=np.int32 )
    
    result = region_of_interest(result, region)
    hough = hough_lines(result, 1, 0.03, 15, 25, 3)
    result = extention_lines(result, hough, region)
    result = weighted_img(result, image)    
    return result

white_output = 'test_videos_output/solidWhiteRight.mp4'
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
# white_output = 'test_videos_output/solidYellowLeft.mp4'
# clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4")
# white_output = 'test_videos_output/challenge.mp4'
# clip1 = VideoFileClip("test_videos/challenge.mp4")
# image = mpimg.imread('test_images/shadows.jpg')
# plt.imshow(process_image(image))
# plt.show() 
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)