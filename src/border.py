import cv2
import numpy as np
from skimage import morphology
from scipy.spatial import ConvexHull
from math import pi

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

#functions from exercises
def compactness_score(mask):
    A = np.sum(mask)

    struct_el = morphology.disk(2)
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    perimeter = mask - mask_eroded
    l = np.sum(perimeter)

    if l == 0:
        return 0

    compactness = (4 * pi * A) / (l ** 2)

    return compactness

def convexity_score(mask):
    coords = np.transpose(np.nonzero(mask))

    if len(coords) < 3:
        return 0

    hull = ConvexHull(coords)

    lesion_area = np.count_nonzero(mask)

    convex_hull_area = hull.volume  # FIX: only volume (area in 2D)

    if convex_hull_area == 0:
        return 0

    convexity = lesion_area / convex_hull_area

    return convexity

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

def extract_border_features(mask):

    return {
        "compactness": compactness_score(mask),
        "convexity": convexity_score(mask)
    }

def border(img_id):

    image_path = '../data/' + "imgs/" + img_id
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    _, _, _, img_clean = removeHair_auto(img, gray)

    gray_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    mask = get_mask(gray_clean)
    mask = cut_mask(mask)
    features = extract_border_features(mask)

    return features