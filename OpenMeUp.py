import cv2 as cv
import numpy as np

def grey_scale(image):
    height, width, _ = image.shape

    grey_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            b, g, r = image[i, j]
            value = int((r * 0.299) + (g * 0.587) + (b * 0.114))
            grey_image[i, j] = value
    
    return grey_image

def threshold(image,threshold_value = 127):
    height, width= image.shape

    threshold_img = np.zeros((height,width),dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r = image[i,j]
            if r > threshold_value:
                threshold_img[i,j] = 255
            else:
                threshold_img[i,j] = 0
    return threshold_img