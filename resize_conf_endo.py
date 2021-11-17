import glob
import os
import cv2

gt=sorted(glob.glob('poprawa conf_endo_micr/gt/*'))
dest_path= 'poprawa conf_endo_micr/gt/'

scr_path_org='poprawa conf_endo_micr/org/*'
org=sorted(glob.glob(scr_path_org))
dest_path_org= 'poprawa conf_endo_micr/resized_org/'

for filepath in org:
    gt_img=cv2.imread(filepath)
    resized=cv2.resize(gt_img,dsize=(0,0), fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    basename = os.path.basename(filepath)
    cv2.imwrite(dest_path_org+'resized_'+basename,resized)

