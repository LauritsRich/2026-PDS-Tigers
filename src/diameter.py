import numpy as np
import cv2
from scipy.spatial.distance import pdist
import pandas as pd
def removeHair_auto(img_org, img_gray):

    kernel_size = 5
    threshold = 10
    radius = 3

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # Detect dark hair
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # tophat to detect white hair
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    combined = cv2.add(blackhat, tophat)

    # Threshold
    _, mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

    # Inpaint
    img_out = cv2.inpaint(img_org, mask, radius, cv2.INPAINT_TELEA)

    return blackhat,tophat,combined,img_out

def cut_mask(mask):
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = np.where(col_sums > 0)[0]
    active_rows = np.where(row_sums > 0)[0]

    if len(active_cols) == 0 or len(active_rows) == 0:
        return mask

    col_min, col_max = active_cols[0], active_cols[-1]
    row_min, row_max = active_rows[0], active_rows[-1]

    return mask[row_min:row_max+1, col_min:col_max+1]

def get_mask(gray):

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    mask = thresh > 0

    return mask
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def diameter_features(mask):
  

    # Get coordinates of lesion pixels
    coords = np.column_stack(np.nonzero(mask))
    if coords.shape[0] < 2:
        return {
            "equiv_diameter": 0,
            "diameter_irregularity": 0
        }

    # --- Area-based features ---
    area = coords.shape[0]
    equiv_diameter = np.sqrt(4 * area / np.pi)

    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    max_diameter = np.sqrt((y_max - y_min)**2 + (x_max - x_min)**2)

    # --- Diameter irregularity ---
    diameter_irregularity = max_diameter / equiv_diameter if equiv_diameter != 0 else 0
    return [diameter_irregularity(mask)]

def diameter(img_id):

    image_path = '../data/' + "imgs/" + img_id
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hair removal
    _, _, _, img_clean = removeHair_auto(img, gray)

    gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

    # Mask
    mask = get_mask(gray_clean)
    mask = cut_mask(mask)

    # Diameter features
    features = diameter_features(mask)

    return features