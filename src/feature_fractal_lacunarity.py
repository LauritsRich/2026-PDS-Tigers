import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CLEAN + CROP MASK
# -----------------------------
data_path = '2026-PDS-Tigers/data/'
def load_mask(image_id, data_path=data_path):
    mask_path = data_path + "masks/"
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")

    
    mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return None
    
    return mask


def preprocess_mask(mask):
    mask = (mask > 127).astype(np.uint8) # Converts the gray-scale mask to binary (0 and 1)

    if np.sum(mask) == 0: # If the mask is black (no lesion), return None
        return None

    coords = np.column_stack(np.where(mask > 0)) # Get the coordinates of the non-zero pixels (the lesion)
    y_min, x_min = coords.min(axis=0) # Get the minimum y and x coordinates (top-left corner of the bounding box)
    y_max, x_max = coords.max(axis=0) # Get the maximum y and x coordinates (bottom-right corner of the bounding box)

    mask = mask[y_min:y_max+1, x_min:x_max+1] # Crop the mask to the bounding box of the lesion

    return mask

# -----------------------------
# FAST LACUNARITY (integral image)
# -----------------------------
def compute_lacunarity(mask, box_size):
    h, w = mask.shape # Get the height and width of the mask

    if h < box_size or w < box_size: # If the mask is smaller than the box size, lacunarity cannot be computed, return 0
        return 0

    integral = cv2.integral(mask) # Creates an Integral Image. This is an optimization trick:
                                  # It allows the code to calculate the sum of pixels in any square box using only 4 math operations, regardless of how big the box is.

    counts = [] # This will store the count of pixels in each box as we slide it across the image

    for i in range(h - box_size + 1): # Slide the box vertically across the image. We stop at h - box_size + 1 to avoid going out of bounds.
        for j in range(w - box_size + 1): # Slide the box horizontally across the image. We stop at w - box_size + 1 to avoid going out of bounds.

            total = (
                integral[i + box_size, j + box_size]
                - integral[i, j + box_size]
                - integral[i + box_size, j]
                + integral[i, j]
            ) # This calculates the sum of lesion pixels in the current box using the integral image. It uses the values at the corners of the box to compute the total.

            counts.append(total)

    counts = np.array(counts) # Convert the list of counts to a NumPy array for easier statistical calculations

    if len(counts) == 0: # If there are no boxes (which can happen if the mask is too small), lacunarity cannot be computed,
        return 0

    mean = np.mean(counts) # Calculate the mean number of lesion pixels per box. This is used in the lacunarity formula.

    if mean == 0: # If the mean is zero, it means all boxes are empty (no lesion pixels), so lacunarity is defined to be 0 in this case.
        return 0

    var = np.var(counts) # Calculate the variance of the number of lesion pixels per box. This is also used in the lacunarity formula.

    return (var / (mean ** 2)) + 1 # The formula for lacunarity.

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def lacunarity_for_masks(image_id, box_size=4):
    

    mask = load_mask(image_id)

    if mask is None:
        return None

    mask = preprocess_mask(mask)

    if mask is None:
        return None

    lac = compute_lacunarity(mask, box_size)


    return lac