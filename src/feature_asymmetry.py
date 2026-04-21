import cv2
import numpy as np
from skimage.transform import rotate

data_path = '2026-PDS-Tigers/data/'

def load_mask(image_id, data_path=data_path):
    mask_path = data_path + "masks/"
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")

    
    mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None

    
    mask = mask > 0
    return mask
def safe_crop(mask):
    y, x = np.nonzero(mask)
    if len(y) == 0 or len(x) == 0:
        return None

    y0, y1 = y.min(), y.max()
    x0, x1 = x.min(), x.max()
    return mask[y0:y1+1, x0:x1+1]
def extract_asymmetry(image_id):
    mask = load_mask(image_id)


    if mask is None:
        return np.nan

    mask = mask.astype(bool)
    segment = safe_crop(mask)

    if segment is None:
        return np.nan

    area = np.sum(segment)
    if area == 0:
        return np.nan

    scores = []

    
    for angle in (0, 45, 90, 135):
        if angle == 0:
            rotated = segment
        else:
            rotated = rotate(
                segment.astype(np.uint8),
                angle=angle,
                order=0,               
                preserve_range=True
            ) > 0.5

        rotated = safe_crop(rotated)
        if rotated is None:
            continue

        area = np.sum(rotated)
        if area == 0:
            continue

        flipped = np.flip(rotated, axis=1)
        xor_area = np.sum(rotated ^ flipped)
        scores.append(xor_area / area)

    return np.mean(scores) if scores else np.nan