import os
import numpy as np
from skimage import io, color, feature

data_path = '2026-PDS-Tigers/data/'

def contrast(img_id):
    levels = 32
    try:
        image_path = os.path.join(data_path, 'imgs', img_id)
        image = io.imread(image_path)
        
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        # Step 1: Convert the images to grayscale:
        if image.ndim == 3: # Checks if the image has three channels(RGB)
            image = color.rgb2gray(image) # Convert to grayscale

        # Step 2: Reducing the matrix size:
        image_32 = (image * (levels - 1)).astype(np.uint8) # This reduces the pixel values to a range of 0 to 31 (for 32 levels)

        # Step 3/4: Defining the Spatial relationships/ Constructing the GLCM (distance of 1 pixel and 4 angles: 0, 45, 90, and 135 degrees):
        GLCM = feature.graycomatrix(image_32, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=levels)

        # Step 5: Calculating the contrast:
        contrast_value = feature.graycoprops(GLCM, prop = 'contrast')[0]
        average_contrast = float(np.mean(contrast_value))
        # Average contrast across the 4 angles
        return average_contrast
    except:
        return np.nan