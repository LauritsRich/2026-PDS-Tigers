import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from statistics import variance
from skimage.color import rgb2hsv, rgb2lab
from scipy.stats import circmean, circvar
from math import sqrt, nan, sin


def safe_variance(values):
    return variance(values) if len(values) > 1 else 0



data_path = '../data/'
def load_image_and_mask(image_id, data_path=data_path):
    ''' Get the array corresponding to the image and the mask. 
    Remove transparency channel from the image array.   

    '''
    
    img_path = data_path + "imgs/"
    mask_path = data_path + "masks/"

    # Load the image/mask
    file_im = img_path + image_id
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")
    im = plt.imread(file_im)
    mask = plt.imread(file_mask)

    # Get rid of the transparency channel
    return im[:, :, :3], mask


def slic_segmentation(image, mask, n_segments = 10, compactness = 0.1):
    '''Get color segments of lesion from SLIC algorithm.
    Optional argument n_segments (default 10) defines desired amount of segments.
    Optional argument compactness (default 0.1) defines balance between color
    and position.

    Args:
        image (numpy.ndarray): image to segment
        mask (numpy.ndarray):  image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position

    Returns:
        slic_segments (numpy.ndarray): SLIC color segments.
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)

    return slic_segments

def get_rgb_means(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = int(np.unique(slic_segments)[-1])

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy()
        segment[slic_segments != i] = -1

        #Get average RGB values from segment
        mask_i = slic_segments == i
        rgb_mean = image[mask_i].mean(axis=0)

        rgb_means.append(rgb_mean)

    return rgb_means

def rgb_var(image, slic_segments):
    '''Get variance of RGB means for each segment in
    SLIC segmentation in red, green and blue channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation

    Returns:
        red_var (float): variance in red channel segment means
        green_var (float): variance in green channel segment means
        blue_var (float): variance in green channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 for area outside mask
        return 0, 0, 0

    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    # Compute variance for each channel seperately
    red_var = safe_variance(red)
    green_var = safe_variance(green)
    blue_var = safe_variance(blue)

    return red_var, green_var, blue_var

def get_hsv_means(image, slic_segments):
    '''Get mean HSV values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        hsv_means (list): HSV mean values for each segment.
    '''

    hsv_image = rgb2hsv(image)

    max_segment_id = int(np.unique(slic_segments)[-1])

    hsv_means = []
    for i in range(1, max_segment_id + 1):

        # Create masked image where only specific segment is active
        segment = hsv_image.copy()
        segment[slic_segments != i] = nan

        #Get average HSV values from segment
        mask_i = slic_segments == i


        hue_mean = circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit') # Compute circular hue mean
        sat_mean = np.mean(segment[:, :, 1][mask_i]) # Compute saturation mean
        val_mean = np.mean(segment[:, :, 2][mask_i]) # Compute value mean

        hsv_mean = np.asarray([hue_mean, sat_mean, val_mean])

        hsv_means.append(hsv_mean)

    return hsv_means

def hsv_var(image, slic_segments):
    '''Get variance of HSV means for each segment in
    SLIC segmentation in hue, saturation and value channels

    Args:
        image (numpy.ndarray): image to compute color variance for
        slic_segments (numpy.ndarray): array containing SLIC segmentation

    Returns:
        hue_var (float): variance in hue channel segment means
        sat_var (float): variance in saturation channel segment means
        val_var (float): variance in value channel segment means.
    '''

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2: # Use 2 since slic_segments also has 0 marking for area outside mask
        return 0, 0, 0

    hsv_means = get_hsv_means(image, slic_segments)
    n = len(hsv_means) # Amount of segments, used later to compute variance

    # Seperate and collect channel means together in lists
    hue = []
    sat = []
    val = []
    for hsv_mean in hsv_means:
        hue.append(hsv_mean[0])
        sat.append(hsv_mean[1])
        val.append(hsv_mean[2])

    # Compute variance for each channel seperately
    hue_var = circvar(hue, high=1, low=0)
    sat_var = safe_variance(sat)
    val_var = safe_variance(val)

    return hue_var, sat_var, val_var

def circular_max_min(hue_means):

    '''Get the difference in means between max hue and min hue. Takes into account that hue is circular'''
    hue = np.array(hue_means)

    diff = np.abs(hue[:, None] - hue[None, :])

    circular_diff = np.minimum(diff, 1 - diff)

    return np.max(circular_diff)


##### NEW FEATURE - L*a*b* format-medical usage
def lab_means_per_segment(image, slic_segments):
    '''Get mean LAB values for each segment in a SLIC segmented image.'''

    lab_image = rgb2lab(image)
    segment_ids = np.unique(slic_segments)
####Remove the background except mask
    segment_ids = segment_ids[segment_ids != 0]

    lab_means = []

    for seg_id in segment_ids:
        mask = slic_segments == seg_id

        # Extract pixels belonging to the segment
        segment_pixels = lab_image[mask]

        # Compute mean LAB
        lab_mean = segment_pixels.mean(axis=0)

        lab_means.append(lab_mean)

    return lab_means

def lab_mean_and_variation(image, slic_segments):
    '''Get overall mean and variance for the slic segments'''

    means_per_segment = np.array(lab_means_per_segment(image, slic_segments))
    L_channel = means_per_segment[:, 0]
    a_channel = means_per_segment[:, 1]
    b_channel = means_per_segment[:, 2]


    L_mean = np.mean(L_channel)
    a_mean = np.mean(a_channel)
    b_mean = np.mean(b_channel)

    L_var = safe_variance(L_channel)
    a_var = safe_variance(a_channel)
    b_var = safe_variance(b_channel)
    return L_mean,a_mean,b_mean, L_var,a_var,b_var


#### Combined function for all possible feature combinations
def color_features_extraction(image_id):
    image, mask = load_image_and_mask(image_id)
    slic_segments = slic_segmentation(image, mask)

    r_var, g_var, b_var = rgb_var(image, slic_segments)

    h_var, s_var, v_var = hsv_var(image, slic_segments)

    hsv_means = get_hsv_means(image, slic_segments)
    hue_means = np.array([hsv[0] for hsv in hsv_means])
    s_means = [hsv[1] for hsv in hsv_means]
    v_means = [hsv[2] for hsv in hsv_means]

    rgb_means = get_rgb_means(image, slic_segments)
    r_means = [rgb[0] for rgb in rgb_means]
    g_means = [rgb[1] for rgb in rgb_means]
    b_means = [rgb[2] for rgb in rgb_means]

    circular_max_min_h = circular_max_min(hue_means)
    ##combine all variances together

    ## Magnitude of variance
    rgb_var_mag = np.sqrt(r_var**2 + g_var**2 + b_var**2)
    hsv_var_mag = np.sqrt(h_var**2 + s_var**2 + v_var**2)

    ## Mean of variance
    rgb_var_mean = np.mean([r_var, g_var, b_var])
    hsv_var_mean = np.mean([h_var, s_var, v_var])

### hue is circular thats why we use trigonometric functions as below:
    h_cos = np.cos(2 * np.pi * hue_means)
    h_sin = np.sin(2 * np.pi * hue_means)

    mean_angle_h = np.arctan2(np.mean(h_sin), np.mean(h_cos))

    v_value = np.mean(v_means)
    s_value = np.mean(s_means)

    r_value = np.mean(r_means)
    g_value = np.mean(g_means)
    b_value = np.mean(b_means)

    ### LAB thingies
    Ls_value, as_value, bs_value, Ls_var, as_var, bs_var = lab_mean_and_variation(image, slic_segments)


    #### keep the hue distance, magnitude of the rgb and hsv variances and the circular; return all posible features to see the ones that affect the most 
    return [Ls_value, as_value, bs_value, mean_angle_h, s_value, v_value, r_value, g_value, b_value,Ls_var, as_var, bs_var, h_var, s_var, v_var, r_var, g_var, b_var, hsv_var_mean, rgb_var_mean, hsv_var_mag, rgb_var_mag, circular_max_min_h]


    






