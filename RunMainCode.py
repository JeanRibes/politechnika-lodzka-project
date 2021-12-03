import glob
import os
import cv2
import numpy as np
from snorkel.labeling import LFApplier
from snorkel.labeling.model import LabelModel
from labeling_fcns import lfs, window_size,h,w
from gold_labels import get_gold_labels
from collections import namedtuple
from pathlib import Path

PixelData = namedtuple('PixelData', ['labels','x','y'])
Picture = namedtuple('Picture', ['pixels', 'filename'])

gt_path='labeling/gt/' # ground truth
dest_path='labeling/masks_generated/' # labelling fnuction output ?
source_path='labeling/source/' # raw images

images_sorted=sorted(glob.glob(source_path+'*'))
#images_splitted=np.array_split(images_sorted, 26) # makes groups of images

source_files = set(Path(x).name for x in glob.glob('labeling/source/*'))
generated_files = set(Path(x).name for x in glob.glob('labeling/masks_generated/*'))

to_process = source_files - generated_files
to_process = [source_path + filename for filename in to_process]
print(to_process)

images_splitted=np.array_split(to_process, 26)

rows=h
cols=w

if False:
    L_train = []
    applier = LFApplier(lfs)
    image = cv2.imread('labeling/source/1_0.png')
    filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    image_pixels=[]
    for x in range(0, rows):
        for y in range(0, cols):
            cutted_image = filtered_image[x:(x + window_size), y:(y + window_size)]
            output_labels = applier.apply([cutted_image], progress_bar=False)
            pixelData=PixelData(output_labels,x,y)
            image_pixels.append(pixelData)
            L_train.append(output_labels[0])
    print("training data")
    label_model = LabelModel(cardinality=2, verbose=False, device='cpu')
    print("fitting data")
    label_model.fit(L_train=np.array(L_train), seed=123,log_freq=1)

    image = np.zeros((rows, cols))
    for pixel in image_pixels:
        pred = label_model.predict(pixel.labels)
        if pred > 0:
            pred = 0
        else:
            pred = 255
        image[pixel.x, pixel.y] = pred
    cv2.imwrite('labeling/masks_generated/1_0.png', image)


exit(0)

L_train = []
applier = LFApplier(lfs)
pictures = []
for filepath in to_process:
    print(f"learning {filepath}")
    filename = os.path.basename(filepath)
    image=cv2.imread(filepath)
    image_pixels = []
    filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    for x in range(0, rows):
        for y in range(0, cols):
            cutted_image= filtered_image[x:(x + window_size), y:(y + window_size)]
            output_labels = applier.apply([cutted_image], progress_bar=False)
            pixelData=PixelData(output_labels,x,y)
            image_pixels.append(pixelData)
            L_train.append(output_labels[0])

    picture=Picture(image_pixels,filepath)
    pictures.append(picture)

    L = np.array(L_train)
for filepath in to_process:
    print("training data")
    label_model = LabelModel(cardinality=2, verbose=False, device='cpu')
    #label_model = LabelModel(cardinality=2, verbose=True,device='cuda')
    print("fitting data")
    label_model.fit(L_train=L, seed=123,log_freq=1)

print("predicting images")
for picture in pictures:
    image=np.zeros((rows,cols))
    for pixel in picture.pixels:
        pred=label_model.predict(pixel.labels)
        if pred>0:
            pred=0
        else:
            pred=255
        image[pixel.x,pixel.y]=pred
    current_path=dest_path+os.path.basename(picture.filename)
    cv2.imwrite(current_path, image)
    print(f"written image {picture.filename}")

if __name__ == '__main__':
    print('bye')