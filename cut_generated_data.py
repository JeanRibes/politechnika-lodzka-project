import glob
import cv2
from source_code.labeling_fcns import h, w, margin

images = sorted(glob.glob('120x220/gt/*'))
rows=h
cols=w

for filepath in images:
    image=cv2.imread(filepath)
    without_window=image[(margin-1):(rows+(margin-1)),(margin-1):cols+(margin-1)]
    cv2.imwrite(filepath,without_window)
