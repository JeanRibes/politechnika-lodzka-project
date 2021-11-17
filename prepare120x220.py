import glob
import os
import cv2
from sklearn.feature_extraction import image as sk_im

h=220
w=120

scr=sorted(glob.glob('120x220/working_org/*'))
dest='120x220/org_plus_20/'

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

for filepath in scr:
    basename = os.path.basename(filepath)
    img=cv2.imread(filepath)

    splitted=os.path.splitext(basename)
    name=splitted[0]
    extension = splitted[1]

    if img.shape[0] < h & img.shape[1]>=h:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print('Rotated h'+' '+basename)

    if img.shape[1] < w & img.shape[0]>=w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print('Rotated h'+' '+basename)

    if img.shape[0]<h:
        img = image_resize(img, height=h)
        print('Resized h' + ' ' + basename)

    if img.shape[1]<w:
        img = image_resize(img, width=w)
        print('Resized w' + ' ' + basename)

    org_patches = sk_im.extract_patches_2d(img, (h, w), 0.0001, 1)
    if (len(org_patches) > 0):
        for idx, patch in enumerate(org_patches):
            cv2.imwrite(dest +name+str('_')+ str(idx) + extension, patch)
            print(dest + name + str('_') + str(idx) + extension)
        os.remove(filepath)
    else:
        org_patches = sk_im.extract_patches_2d(img, (h, w), 0.001, 1)
        if (len(org_patches) > 0):
            for idx, patch in enumerate(org_patches):
                cv2.imwrite(dest + name+str('_')+ str(idx) + extension, patch)
            os.remove(filepath)
        else:
            org_patches = sk_im.extract_patches_2d(img, (h, w), 0.01, 1)
            if (len(org_patches) > 0):
                for idx, patch in enumerate(org_patches):
                    cv2.imwrite(dest + name+str('_')+ str(idx) + extension, patch)
                os.remove(filepath)
            else:
                org_patches = sk_im.extract_patches_2d(img, (h, w), 0.1, 1)
                if (len(org_patches) > 0):
                    for idx, patch in enumerate(org_patches):
                        cv2.imwrite(dest + name+str('_')+ str(idx) + extension, patch)
                else:
                    org_patches = sk_im.extract_patches_2d(img, (h, w), 1, 1)
                    if (len(org_patches) > 0):
                        for idx, patch in enumerate(org_patches):
                            cv2.imwrite(dest + name + str('_') + str(idx) + extension, patch)
                        os.remove(filepath)
                    print('Patches not created from: ' + filepath)

