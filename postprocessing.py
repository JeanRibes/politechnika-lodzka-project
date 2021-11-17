import glob
import os
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import pandas as pd

gt_path = 'F:\\obrazy\\gt\\'
dataset_0_masks = 'F:\\obrazy\\bez masek\\preds\\*'
dataset_50_masks = 'F:\\obrazy\\50masek\\preds\\*'
dataset_100_masks = 'F:\\obrazy\\100masek\\preds\\*'

dataset_50_seklet = 'F:\\obrazy\\50masek\\skelet\\'
dataset_100_skelet = 'F:\\obrazy\\100masek\\skelet\\'
dataset_50_seklet_dil = 'F:\\obrazy\\50masek\\skelet+dilation+close\\'
dataset_100_skelet_dil = 'F:\\obrazy\\100masek\\skelet+dilation+close\\'

dataset_0_seklet = 'F:\\obrazy\\bez masek\\skelet\\'
dataset_0_skelet_dil = 'F:\\obrazy\\bez masek\\skelet+dilation+close\\'

dataset_50_thresholded = 'F:\\obrazy\\50masek\\thresholded\\'
dataset_0_thresholded='F:\\obrazy2\\bez masek\\thresholded\\'
dataset_100_thresholded='F:\\obrazy2\\100masek\\thresholded\\'

def calc_dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def calc_hf(one, two):
    return directed_hausdorff(one, two)[0]

def do_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh[thresh > 0] = 1
    return thresh

def do_dilation(img):
    skeleton_lee = do_threshold(img)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(skeleton_lee, kernel)
    return dilation

def get_df(path):
    names = []
    dices = []
    hds = []

    dataset = sorted(glob.glob(path))

    for list_idx, filepath in enumerate(dataset):
        filename = os.path.basename(filepath)
        names.append(filename)

        img = cv2.imread(filepath)
        img_postprocessed = do_dilation(img)

        gt = cv2.imread(gt_path + filename)
        gt_postprocessed = do_dilation(gt)

        dice_val = calc_dice(gt_postprocessed, img_postprocessed)
        dices.append(dice_val)

        img_skelet = do_threshold(img)
        gt_skelet = do_threshold(gt)

        hd = calc_hf(img_skelet,gt_skelet)
        hds.append(hd)

    df = pd.DataFrame({'Nazwa pliku': names, 'DICE': dices, 'Hausdorff': hds})
    return df

def prepare_skel_dil_imgs(dataset,dest_skelet_dil):
    for list_idx, filepath in enumerate(dataset):
        filename = os.path.basename(filepath)
        img = cv2.imread(filepath)
        img_process = do_dilation(img)
        cv2.imwrite(dest_skelet_dil + filename, img_process)

dataset50 = sorted(glob.glob(dataset_50_masks))
dataset100=sorted(glob.glob(dataset_100_masks))
dataset0=sorted(glob.glob(dataset_0_masks))

prepare_skel_dil_imgs(dataset50,dataset_50_seklet_dil)
prepare_skel_dil_imgs(dataset100,dataset_100_skelet_dil)
prepare_skel_dil_imgs(dataset0,dataset_0_thresholded)

writer = pd.ExcelWriter('results.xlsx', engine='openpyxl')

get_df(dataset_0_masks).to_excel(writer, sheet_name='0%')
get_df(dataset_50_masks).to_excel(writer, sheet_name='50%')
get_df(dataset_100_masks).to_excel(writer, sheet_name='100%')

writer.save()
writer.close()
