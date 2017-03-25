#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of x steps:
* Grayscale conversion - example conversion from cv2 library.
* Histogram equalization - using CLAHE alghoritm, used to enhance contrast of the image.
* Gaussian blur - from examples, used for softening the edges.
* Canny converion - from examples, edge detection.
* Region of interest masking - trapezoidal shape was used with flexible size (percentage of image dimensions).
* Hough lines detection - coefficients tuned up on all of sample images.
* Line detection and extending (custom function), it consists of steps listed below:
* * coefficient filtering - remove all detected lines with slope coefficient lower than 0.4
* * side detection - based on slope cooeficient sign
* * obtaining line equation - 1st degree polynomial fit was used to find line equation. To smooth the line movement the mean window filter of size 4 was applied.
* * obtaining line bor der points: mask region was uzed to calculate the terminal points of the line.
* Image joining - original image and lines were joined to visualize the effect.


### 2. Identify potential shortcomings with your current pipeline

Obtaining line equation by polynomial fit is good when data filtering is higly good. A feq additional points poorly detected on a screen can disturb final results. 

### 3. Suggest possible improvements to your pipeline

For better error handling, and false positive line results the parameters of the filters can be tuned up. Also additional methods of filtering of data could be applied.

Mean window filtering for line smoothing can be altered with for example median filtering. Also some experiments with the mean window could be done.