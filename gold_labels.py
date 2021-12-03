import os
import cv2
import numpy as np

def get_gold_labels(images_list,gt_path):
    """
    extracts an image pixel by pixel
    """
    gold_labels=[]
    for idx,filepath in enumerate(images_list):
        filename = os.path.basename(filepath)
        print('gold_labels: loading',gt_path+filename)
        image=cv2.imread(gt_path+filename)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width,_ = image.shape

        for x in range(0, height):
            for y in range(0, width):
                label=gray[x,y]
                label= 0 if label > 0 else 1
                gold_labels.append(label)
    
        del image
        del gray
    return np.array(gold_labels)
