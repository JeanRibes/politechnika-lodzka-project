import os
import cv2
import numpy as np

def get_gold_labels(list,gt_path):
    gold_labels=[]
    for idx,filepath in enumerate(list):
        filename = os.path.basename(filepath)
        image=cv2.imread(gt_path+filename)
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = image.shape[0]
        width = image.shape[1]

        for x in range(0, height):
            for y in range(0, width):
                label=gray[x,y]
                label= 0 if label > 0 else 1
                gold_labels.append(label)

    return np.array(gold_labels)