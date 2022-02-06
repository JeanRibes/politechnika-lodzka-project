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
    laplacian = laplacian - laplacian.min()
    laplacian = laplacian / laplacian.max() * 255
    laplacian = np.uint8(laplacian)
    a, laplacian = cv2.threshold(laplacian, 140, 255, cv2.THRESH_BINARY)
    closing = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return opening


@labeling_function(preprocessors=[to_gray, _laplacian_lf])
def laplacian_lf(image, x, y):
    return pixel_decision(image[x, y])


def _jean(image):
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 0)
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, np.ones((4, 4), np.uint8))

@labeling_function(preprocessors=[to_gray, _jean])
def jean(image,x,y):
    return pixel_decision(image[x,y])


def _watershed(image):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 0)

    inv = cv2.bitwise_not(thresh)
    dist_transform = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(inv, np.ones((3, 3), np.uint8), iterations=3)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

@labeling_function(preprocessors=[_watershed])
def watershed_segmentation(image,x,y):
    if image[x,y] > 0.75 * 255:
        return EDGE
    else:
        return ABSTAIN
# pas fini car la sortie es pas correcte !

labeling_functions = [
    adaptiveThreshold_gaussian_lf,
    adaptiveThreshold_mean_lf,
    #global_thresholding,
    laplacian_lf,
    jean,
    watershed_segmentation
]
#_original_labeling_functions = [
#    adaptiveThreshold_gaussian_lf,
#    adaptiveThreshold_mean_lf,
#    global_thresholding,
#    laplacian_lf,
#]