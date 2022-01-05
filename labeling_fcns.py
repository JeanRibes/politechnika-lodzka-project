import numpy as np
from snorkel.labeling import labeling_function
import cv2

window_size=9
margin=10
blockSize=window_size*window_size
middle_pixel=int((window_size-1)/2)
#h=200
#w=100
h=220
w=120

from snorkel.preprocess import preprocessor
@preprocessor(memoize=True)
def to_gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

ABSTAIN = -1
EDGE=1
CELL=0
def getMiddlePixel(x):
    m_x, m_y = x.shape
    m_x = int(m_x/2)
    m_y = int(m_y/2)
    y = x[m_x, m_y]

    if y > 0.75*255:
        return EDGE
    elif y < 0.25*255:
        return CELL
    else:
        return ABSTAIN
    #return 0 if y>0 else 1

@labeling_function(pre=[to_gray])
def adaptiveThreshold_gaussian_lf(x):
    thresh = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, window_size,1)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    thresh_erode = cv2.morphologyEx(closing, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    return getMiddlePixel(thresh_erode)

@labeling_function(pre=[to_gray])
def adaptiveThreshold_mean_lf(x):
    thresh = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, window_size,0)
    thresh_erode = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, np.ones((3, 3), np.uint8))
    return getMiddlePixel(thresh_erode)

@labeling_function(pre=[to_gray])
def global_thresholding(x):
    ret, global_thresholding = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_erode = cv2.morphologyEx(global_thresholding, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    return getMiddlePixel(thresh_erode)

@labeling_function(pre=[to_gray])
def laplacian_lf(x):
    laplacian = cv2.Laplacian(x, cv2.CV_64F, ksize=11)
    closing = cv2.morphologyEx(laplacian, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    erode = cv2.morphologyEx(closing,cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return getMiddlePixel(opening)

lfs = [adaptiveThreshold_gaussian_lf, global_thresholding,laplacian_lf,adaptiveThreshold_mean_lf]

