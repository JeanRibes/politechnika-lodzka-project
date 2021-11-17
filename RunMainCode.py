import glob
import os
import cv2
import numpy as np
from snorkel.labeling import LFApplier
from snorkel.labeling.model import LabelModel
from source_code.Picture import Picture
from source_code.PixelData import PixelData
from source_code.labeling_fcns import lfs, window_size,h,w
from source_code.gold_labels import get_gold_labels

gt_path='labeling/gt/'
dest_path='labeling/masks_generated/'
source_path='labeling/source/*'

images_sorted=sorted(glob.glob(source_path))
images_splitted=np.array_split(images_sorted, 26)

rows=h
cols=w

for list_idx,list in enumerate(images_splitted):

        df_train_Pictures = []
        L_train = []
        applier = LFApplier(lfs)

        gold=get_gold_labels(list,gt_path)
        for idx,filepath in enumerate(list):
            filename = os.path.basename(filepath)
            image=cv2.imread(filepath)
            cv2.namedWindow('gray_scale', cv2.WINDOW_NORMAL)
            cv2.imshow('gray_scale', image)
            cv2.waitKey(0)
            image_pixels = []
            filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
            for x in range(0, rows):
                for y in range(0, cols):
                    cutted_image= filtered_image[x:(x + window_size), y:(y + window_size)]
                    output_labels = applier.apply([cutted_image])
                    pixelData=PixelData(output_labels,x,y)
                    image_pixels.append(pixelData)
                    L_train.append(output_labels[0])

            pic=Picture(image_pixels,filepath)
            df_train_Pictures.append(pic)

        L = np.array(L_train)

        label_model = LabelModel(cardinality=2, verbose=True,device='cuda')
        label_model.fit(L_train=L, seed=123,log_freq=1)

        for picture in df_train_Pictures:
            image=np.zeros((rows,cols))
            for idx2, pixel in enumerate(picture.pixels):
                pred=label_model.predict(pixel.labels)
                if pred>0:
                    pred=0
                else:
                    pred=255
                image[pixel.x,pixel.y]=pred
            current_path=dest_path+os.path.basename(picture.filename)
            cv2.imwrite(current_path, image)