import cv2 as cv
import numpy as np

im = cv.imread("IMG/pic1.webp")
grayscale = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

_, threshold = cv.threshold(grayscale, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = np.ones((5,5), np.uint8)

closed = cv.morphologyEx(threshold, cv.MORPH_DILATE, kernel)
cv.imwrite("Result1/e.webp",closed)

# Find contours of blobs
contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if(cnt.size>0):
        x, y, w, h = cv.boundingRect(cnt)
        cv.drawContours(im, [cnt], -1, (0, 0, 255), 1)
        cv.rectangle(im, (x, y), (x+w, y+h), (255, 255, 0), 1)

cv.imwrite("Result1/highlighted_cells.webp", im)