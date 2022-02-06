import glob
import os
import time

import cv2
import numpy as np
from snorkel.labeling.model import LabelModel
from labeling_fcns import h, w
from collections import namedtuple
from pathlib import Path

from norkl import Applier
from labeling_functions2 import labeling_functions
PixelData = namedtuple('PixelData', ['labels', 'x', 'y'])
Picture = namedtuple('Picture', ['pixels', 'filename'])

gt_path = 'labeling/gt/'  # ground truth
dest_path = 'labeling/masks_generated/'  # labelling fnuction output ?
source_path = 'labeling/source/'  # raw images

images_sorted = sorted(glob.glob(source_path + '*'))
# images_splitted=np.array_split(images_sorted, 26) # makes groups of images

source_files = set(Path(x).name for x in glob.glob('labeling/source/*'))
generated_files = set(Path(x).name for x in glob.glob('labeling/masks_generated/*'))

#to_process = source_files - generated_files
to_process = sorted(list(source_files))
to_process = [source_path + filename for filename in to_process]
to_process.sort()
#to_process = to_process[0:1]
#to_process = ['labeling/source/1_0.png']
print(to_process)

rows = h
cols = w

L_train = []
applier = Applier(labeling_functions)
pictures = []

for filepath in to_process[0:1]:
    print(f"learning {filepath}")
    filename = os.path.basename(filepath)
    image = cv2.imread(filepath)
    image_pixels = []
    filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    for x in range(0, rows):
        for y in range(0, cols):
            output_labels = np.array(applier.apply(filtered_image, x, y))
            pixelData = PixelData(output_labels, x, y)
            image_pixels.append(pixelData)
            L_train.append(output_labels[0])

    picture = Picture(image_pixels, filepath)
    pictures.append(picture)

label_model = LabelModel(cardinality=2, verbose=False, device='cpu')
L = np.array(L_train)
label_model.fit(L_train=L, seed=123, log_freq=1)

print("predicting images")
for picture in pictures:
    image = np.zeros((rows, cols))
    for pixel in picture.pixels:
        pred = label_model.predict(pixel.labels)
        if pred > 0:
            pred = 255
        else:
            pred = 0
        image[pixel.x, pixel.y] = pred
    current_path = dest_path + os.path.basename(picture.filename)
    cv2.imwrite(current_path, image)
    print(f"written image {current_path}")

if __name__ == '__main__':
    print('bye')

