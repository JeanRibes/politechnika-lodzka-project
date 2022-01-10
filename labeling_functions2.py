import numpy as np
import cv2
from norkl import labeling_function

window_size=9

ABSTAIN = -1
EDGE = 1
CELL = 0

# def to_gray(x):
#    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
to_gray = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def pixel_decision(pixel):
    if pixel > 0.75 * 255:
        return EDGE
    elif pixel < 0.25 * 255:
        return CELL
    else:
        return ABSTAIN


def _adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window_size, 1)


@labeling_function(preprocessors=[
    to_gray,
    _adaptive_threshold,
    lambda image: cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
    lambda image: cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8)),
])
def adaptiveThreshold_gaussian_lf(image, x, y):
    return pixel_decision(image[x, y])


@labeling_function(preprocessors=[
    to_gray,
    lambda image: cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, window_size, 0),
    lambda image: cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
])
def adaptiveThreshold_mean_lf(image, x, y):
    return pixel_decision(image[x, y])


@labeling_function(preprocessors=[
    to_gray,
    lambda image: cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    lambda image: cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
])
def global_thresholding(image, x, y):
    return pixel_decision(image[x, y])


def _laplacian_lf(x):
    laplacian = cv2.Laplacian(x, cv2.CV_64F, ksize=11)
    closing = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return opening


@labeling_function(preprocessors=[to_gray, _laplacian_lf])
def laplacian_lf(image, x, y):
    return pixel_decision(image[x, y])

labeling_functions = [
    adaptiveThreshold_gaussian_lf,
    adaptiveThreshold_mean_lf,
    global_thresholding,
    laplacian_lf
]