'''
    Written in Python 3.5
    https://github.com/CODAIT/deep-histopath/blob/master/deephistopath/preprocessing.py
'''

import time
import random
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import math
import gc
import pdb
import datetime
from enum import Enum

import openslide
from openslide import OpenSlide, OpenSlideUnsupportedFormatError,OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

from scipy.ndimage.morphology import binary_fill_holes

from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from skimage.filters import threshold_otsu, gaussian
import skimage.measure
from skimage.transform import rescale, rotate
from scipy.stats import mode
from scipy import ndimage

import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation

from itertools import chain
import matplotlib.pylab as plt


ADDITIONAL_NP_STATS = False
# PATCH_SIZE = 64
CHANNEL = 3
THRESH = 90
PIXEL_WHITE = 255
PIXEL_TH = 200
PIXEL_BLACK = 0

SCALE_FACTOR = 32

TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

HSV_PURPLE = 270
HSV_PINK = 330

# level = 3
# verbose = 0

def construct_colored_wsi(rgba_):

    '''
        This function splits and merges R, G, B channels.
        HSV and GRAY images are also created for future segmentation procedure.
        Args:
            - rgba_: Image to be processed, NumPy array type.
    '''
    r_, g_, b_, a_ = cv2.split(rgba_)
    
    wsi_rgb_ = cv2.merge((r_, g_, b_))
    
    wsi_gray_ = cv2.cvtColor(wsi_rgb_,cv2.COLOR_RGB2GRAY)
    wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2HSV)
    
    return wsi_rgb_, wsi_gray_, wsi_hsv_

'''
    Extract Valid patches.
'''
def construct_bags(wsi_, wsi_rgb, contours, mask, level, PATCH_SIZE):
    
    '''
    Args:
        To-do.
    Returns: 
        - patches: lists of patches in numpy array: [PATCH_WIDTH, PATCH_HEIGHT, CHANNEL]
        - patches_coords: coordinates of patches: (x_min, y_min). The bouding box of the patch
        is (x_min, y_min, x_min + PATCH_WIDTH, y_min + PATCH_HEIGHT)
    '''

    patches = []
    patches_coords = []

    start = time.time()
    
    '''
        !!! 
        Currently we select only the first 5 regions, because there are too many small areas and 
        too many irrelevant would be selected if we extract patches from all regions.
        And how many regions from which we decide to extract patches is 
        highly related to the SEGMENTATION results.
    '''
    contours_ = sorted(contours, key = cv2.contourArea, reverse = True)
    contours_ = contours_[:5] if(len(contours_) >5) else contours_ #edit PC- May 3,2020

    for i, box_ in enumerate(contours_):

        box_ = cv2.boundingRect(np.squeeze(box_))
#         print('region', i)
        
        '''
        !!! Take care of difference in shapes:
            Coordinates in bounding boxes: (WIDTH, HEIGHT)
            WSI image: (HEIGHT, WIDTH, CHANNEL)
            Mask: (HEIGHT, WIDTH, CHANNEL)
        '''

        b_x_start = int(box_[0])
        b_y_start = int(box_[1])
        b_x_end = int(box_[0]) + int(box_[2])
        b_y_end = int(box_[1]) + int(box_[3])
        
        '''
            !!!
            step size could be tuned for better results.
        '''

        X = np.arange(b_x_start, b_x_end,step=PATCH_SIZE -1 ) 
        Y = np.arange(b_y_start, b_y_end, step=PATCH_SIZE -1)  
        # step_size = overlap = half of the image
        # Original =step=PATCH_SIZE // 2
        # Edited PC July 2,2020= step=PATCH_SIZE -1
        
              
        
#         print('ROI length:', len(X), len(Y))
        
        for h_pos, y_height_ in enumerate(Y):
        
            for w_pos, x_width_ in enumerate(X):

                # Read again from WSI object wastes tooooo much time.
                # patch_img = wsi_.read_region((x_width_, y_height_), level, (PATCH_SIZE, PATCH_SIZE))
                
                '''
                    !!! Take care of difference in shapes
                    Here, the shape of wsi_rgb is (HEIGHT, WIDTH, CHANNEL)
                    the shape of mask is (HEIGHT, WIDTH, CHANNEL)
                '''
                patch_arr = wsi_rgb[y_height_: y_height_ + PATCH_SIZE,\
                                    x_width_:x_width_ + PATCH_SIZE,:]            
#                 print("read_region (scaled coordinates): ", x_width_, y_height_)

                width_mask = x_width_
                height_mask = y_height_                
                
                patch_mask_arr = mask[height_mask: height_mask + PATCH_SIZE, \
                                      width_mask: width_mask + PATCH_SIZE]

#                 print("Numpy mask shape: ", patch_mask_arr.shape)
#                 print("Numpy patch shape: ", patch_arr.shape)

                try:
                    bitwise_ = cv2.bitwise_and(patch_arr, patch_mask_arr)
                
                except Exception as err:
                    print('Out of the boundary')
                    pass
                    
#                 f_ = ((patch_arr > PIXEL_TH) * 1)
#                 f_ = (f_ * PIXEL_WHITE).astype('uint8')

#                 if np.mean(f_) <= (PIXEL_TH + 40):
#                     patches.append(patch_arr)
#                     patches_coords.append((x_width_, y_height_))
#                     print(x_width_, y_height_)
#                     print('Saved\n')

                else:
                    bitwise_grey = cv2.cvtColor(bitwise_, cv2.COLOR_RGB2GRAY)
                    white_pixel_cnt = cv2.countNonZero(bitwise_grey)

                    '''
                        Patches whose valid area >= 25% of total area is considered
                        valid and selected.(25)
                    '''

                    if white_pixel_cnt >= ((PATCH_SIZE ** 2) * 0.95): 
                        # changed from 0.25 to 0.90 
                        # edited PC July 02,2020 - Changed to 0.95
                        
                        if patch_arr.shape == (PATCH_SIZE, PATCH_SIZE, CHANNEL):
                            patches.append(patch_arr)
                            patches_coords.append((x_width_, y_height_))
#                           print(x_width_, y_height_)
#                             print('Saved\n')

#                     else:
#                         print('Did not save\n')

    end = time.time()
    print("Time spent on patch extraction: ",  (end - start))

    # patches_ = [patch_[:,:,:3] for patch_ in patches] 
    print("Total number of patches extracted:", len(patches))
    
    return patches, patches_coords

def estimate_blur(image, threshold = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)



def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.

    Shape (h, w, c) to (h, w).

    Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

    Returns:
    Grayscale image as NumPy array with shape (h, w).
    """
    t = Time()
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    np_info(grayscale, "Gray", t.elapsed())
    return grayscale

def filter_rgb_to_hsv(np_img, display_np_info=True):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).

    Args:
    np_img: RGB image as a NumPy array.
    display_np_info: If True, display NumPy array info and filter time.

    Returns:
    Image as NumPy array in HSV representation.
    """

    if display_np_info:
        t = Time()
    hsv = sk_color.rgb2hsv(np_img)
    if display_np_info:
        np_info(hsv, "RGB to HSV", t.elapsed())
    return hsv

def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
    """
    Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
    values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
    https://en.wikipedia.org/wiki/HSL_and_HSV

    Args:
    hsv: HSV image as a NumPy array.
    output_type: Type of array to return (float or int).
    display_np_info: If True, display NumPy array info and filter time.

    Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
    """
    if display_np_info:
        t = Time()
    h = hsv[:, :, 0]
    h = h.flatten()
    if output_type == "int":
        h *= 360
        h = h.astype("int")
    if display_np_info:
        np_info(hsv, "HSV to H", t.elapsed())
    return h

def filter_hsv_to_v(hsv):
    """
    Experimental HSV to V (value).

    Args:
    hsv:  HSV image as a NumPy array.

    Returns:
    Value values as a 1-dimensional NumPy array.
    """
    v = hsv[:, :, 2]
    v = v.flatten()
    return v

def filter_hsv_to_s(hsv):
    """
    Experimental HSV to S (saturation).

    Args:
    hsv:  HSV image as a NumPy array.

    Returns:
    Saturation values as a 1-dimensional NumPy array.
    """
    s = hsv[:, :, 1]
    s = s.flatten()
    return s



def get_contours(cont_img, rgb_image_shape):
    
    '''
    Args:
        - cont_img: images with contours, these images are in np.array format.
        - rgb_image_shape: shape of rgb image, (HEIGHT, WIDTH).
    Returns: 
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of valid region coordinates (contours squeezed);
        - contours: List of valid regions (coordinates);
        - mask: binary mask array;
        !!! It should be noticed that the shape of mask array is: (HEIGHT, WIDTH, CHANNEL).
    '''
    
    print('contour image: ',cont_img.shape)
    
    contour_coords = []
    contours, hiers = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # print(contours)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]

    for contour in contours:
        contour_coords.append(np.squeeze(contour))
        
    mask = np.zeros(rgb_image_shape, np.uint8)
    
    print('mask shape', mask.shape)
    cv2.drawContours(mask, contours, -1, \
                    (PIXEL_WHITE, PIXEL_WHITE, PIXEL_WHITE),thickness=-1)
    
    return boundingBoxes, contour_coords, contours, mask

def hsv_purple_deviation(hsv_hues):
    """
    Obtain the deviation from the HSV hue for purple.
    Args:
    hsv_hues: NumPy array of HSV hue values.
    Returns:
    The HSV purple deviation.
    """
    purple_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PURPLE) ** 2))
    return purple_deviation


def hsv_pink_deviation(hsv_hues):
    """
    Obtain the deviation from the HSV hue for pink.
    Args:
    hsv_hues: NumPy array of HSV hue values.
    Returns:
    The HSV pink deviation.
    """
    pink_deviation = np.sqrt(np.mean(np.abs(hsv_hues - HSV_PINK) ** 2))
    return pink_deviation

def hsv_purple_pink_factor(rgb):
    """
    Compute scoring factor based on purple and pink HSV hue deviations and degree to which a narrowed hue color range
    average is purple versus pink.
    Args:
    rgb: Image an NumPy array.
    Returns:
    Factor that favors purple (hematoxylin stained) tissue over pink (eosin stained) tissue.
    """
    hues = rgb_to_hues(rgb)
    hues = hues[hues >= 260]  # exclude hues under 260
    hues = hues[hues <= 340]  # exclude hues over 340
    if len(hues) == 0:
        return 0  # if no hues between 260 and 340, then not purple or pink
    pu_dev = hsv_purple_deviation(hues)
    pi_dev = hsv_pink_deviation(hues)
    avg_factor = (340 - np.average(hues)) ** 2

    if pu_dev == 0:  # avoid divide by zero if tile has no tissue
        return 0

    factor = pi_dev / pu_dev * avg_factor
    return factor


def hsv_saturation_and_value_factor(rgb):
    """
    Function to reduce scores of tiles with narrow HSV saturations and values since saturation and value standard
    deviations should be relatively broad if the tile contains significant tissue.
    Example of a blurred tile that should not be ranked as a top tile:
    ../data/tiles_png/006/TUPAC-TR-006-tile-r58-c3-x2048-y58369-w1024-h1024.png
    Args:
    rgb: RGB image as a NumPy array
    Returns:
    Saturation and value factor, where 1 is no effect and less than 1 means the standard deviations of saturation and
    value are relatively small.
    """
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    s = filter_hsv_to_s(hsv)
    v = filter_hsv_to_v(hsv)
    s_std = np.std(s)
    v_std = np.std(v)
    if s_std < 0.05 and v_std < 0.05:
        factor = 0.4
    elif s_std < 0.05:
        factor = 0.7
    elif v_std < 0.05:
        factor = 0.7
    else:
        factor = 1

    factor = factor ** 2
    return factor

    
def mask_rgb(rgb, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
    Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.
    Returns:
    NumPy array representing an RGB image with mask applied.
    """
    t = Time()
    result = rgb * np.dstack([mask, mask, mask])
    np_info(result, "Mask RGB", t.elapsed())
    return result

def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
    Args:
    np_img: Image as a NumPy array.
    Returns:
    The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage

def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    if ADDITIONAL_NP_STATS is False:
        print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
    else:
        # np_arr = np.asarray(np_arr)
        max = np_arr.max()
        min = np_arr.min()
        mean = np_arr.mean()
        is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
        print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
          name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))

def open_slide(filepath=None):
    slide = openslide.OpenSlide(filepath)
    
    return slide

def is_purple(crop):
	pooled = skimage.measure.block_reduce(crop, (int(crop.shape[0]/15), int(crop.shape[1]/15), 1), np.average)
	num_purple_squares = 0
	for x in range(pooled.shape[0]):
		for y in range(pooled.shape[1]):
			r = pooled[x, y, 0]
			g = pooled[x, y, 1]
			b = pooled[x, y, 2]
			if is_purple_dot(r, g, b):
				num_purple_squares += 1
	if num_purple_squares > 200: 
		return True,num_purple_squares
	return False,num_purple_squares

def is_purple_dot(r, g, b):
	rb_avg = (r+b)/2
	if r > g - 10 and b > g - 10 and rb_avg > g + 20:
		return True
	return False

def read_wsi(tif_file_path,level= None):
    
    '''
        Identify and load slides.
        Returns:
            - wsi_image: OpenSlide object.
            - rgba_image: WSI image loaded, NumPy array type.
    '''
    
    time_s = time.time()
    
    try:
        wsi_image = OpenSlide(tif_file_path)
        if(level is None):
            level = wsi_image.get_best_level_for_downsample(SCALE_FACTOR)
#         level = wsi_image.level_count-1
        
        slide_w_, slide_h_ = wsi_image.level_dimensions[level]
        
        '''
            The read_region loads the target area into RAM memory, and
            returns an Pillow Image object.
            !! Take care because WSIs are gigapixel images, which are could be 
            extremely large to RAMs.
            Load the whole image in level < 3 could cause failures.
        '''

        # Here we load the whole image from (0, 0), so transformation of coordinates 
        # is not skipped.

        rgba_image_pil = wsi_image.read_region((0, 0), level, (slide_w_, slide_h_))
        print("width, height:", rgba_image_pil.size)

        '''
            !!! It should be noted that:
            1. np.asarray() / np.array() would switch the position 
            of WIDTH and HEIGHT in shape.
            Here, the shape of $rgb_image_pil is: (WIDTH, HEIGHT, CHANNEL).
            After the np.asarray() transformation, the shape of $rgb_image is: 
            (HEIGHT, WIDTH, CHANNEL).
            2. The image here is RGBA image, in which A stands for Alpha channel.
            The A channel is unnecessary for now and could be dropped.
        '''
        rgba_image = np.asarray(rgba_image_pil)
        print("transformed:", rgba_image.shape)
        
    except OpenSlideUnsupportedFormatError:
        print('Exception: OpenSlideUnsupportedFormatError')
        return None

    time_e = time.time()
    
    print("Time spent on loading", tif_file_path, ": ", (time_e - time_s))
    
    return wsi_image, rgba_image, (slide_w_, slide_h_)


def rgb_to_hues(rgb):
    """
    Convert RGB NumPy array to 1-dimensional array of hue values (HSV H values in degrees).
    Args:
    rgb: RGB image as a NumPy array
    Returns:
    1-dimensional array of hue values in degrees
    """
    hsv = filter_rgb_to_hsv(rgb, display_np_info=False)
    h = filter_hsv_to_h(hsv, display_np_info=False)
    return h


def slide_to_scaled_pil_image(filepath=None):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    Args:
    slide_number: The slide number.
    Returns:
    Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    filename = filepath.split("/")[-1].split(".")[0]
    print("Opening Slide",filename)
    #slide = open_slide(slide_filepath)
    slide = open_slide(filepath)
    #slide = openslide(slide_filepath)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / SCALE_FACTOR)
    new_h = math.floor(large_h / SCALE_FACTOR)
    level = slide.get_best_level_for_downsample(SCALE_FACTOR)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h

def small_to_large_mapping(small_pixel, large_dimensions,scale_factor=None):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
    small_pixel: The scaled-down width and height.
    large_dimensions: The width and height of the original whole-slide image.
    Returns:
    Tuple consisting of the scaled-up width and height.
    """
    if(scale_factor is None):
        scale_factor = SCALE_FACTOR
        
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = int(round((large_w / scale_factor) / math.floor(large_w / scale_factor) * (scale_factor * small_x)))
    large_y = int(round((large_h / scale_factor) / math.floor(large_h / scale_factor) * (scale_factor * small_y)))
    return large_x, large_y



def show_slide(slide_number):
    """
    Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.
    Args:
    slide_number: The slide number.
    """
    pil_img = slide_to_scaled_pil_image(slide_number)[0]
    fig = plt.figure(figsize =(20,20))
    plt.imshow(pil_img)
    plt.show()

    
def segmentation_hsv(wsi_hsv_, wsi_rgb_, thresh = None):
    '''
    This func is designed to remove background of WSIs. 
    Args:
        - wsi_hsv_: HSV images.
        - wsi_rgb_: RGB images.
    Returns: 
        - bounding_boxs: List of regions, region: (x, y, w, h);
        - contour_coords: List of arrays. Each array stands for a valid region and 
        contains contour coordinates of that region.
        - contours: Almost same to $contour_coords;
        - mask: binary mask array;
        !!! It should be noticed that:
        1. The shape of mask array is: (HEIGHT, WIDTH, CHANNEL);
        2. $contours is unprocessed format of contour list returned by OpenCV cv2.findContours method.
        
        The shape of arrays in $contours is: (NUMBER_OF_COORDS, 1, 2), 2 stands for x, y;
        The shape of arrays in $contour_coords is: (NUMBER_OF_COORDS, 2), 2 stands for x, y;
        The only difference between $contours and $contour_coords is in shape.
    '''
    print("HSV segmentation: ")
    contour_coord = []
    
    '''
        Here we could tune for better results.
        Currently 20 and 200 are lower and upper threshold for H, S, V values, respectively. 
    
        !!! It should be noted that the threshold values here highly depends on the dataset itself.
        Thresh value could vary a lot among different datasets.
    '''
    lower_ = np.array([20,20,20])
    upper_ = np.array([200,200,200]) 

    # HSV image threshold
    if(thresh is None):
        
        thresh = cv2.inRange(wsi_hsv_, lower_, upper_)
    
    try:
        print("thresh shape:", thresh.shape)
    except:
        print("thresh shape:", thresh.size)
    else:
        pass
    
    '''
        Closing
    '''
    print("Closing step: ")
    close_kernel = np.ones((15, 15), dtype=np.uint8) 
    image_close = cv2.morphologyEx(np.array(thresh),cv2.MORPH_CLOSE, close_kernel)
    print("image_close size", image_close.shape)

    '''
        Openning
    ''' 
    print("Openning step: ")
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
    print("image_open size", image_open.size)

    print("Getting Contour: ")
    bounding_boxes, contour_coords, contours, mask \
    = get_contours(np.array(image_open), wsi_rgb_.shape)
      
    return bounding_boxes, contour_coords, contours, mask



def score_tile(np_tile,tissue_percent):#, row, col):
    """
    Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.
    Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.
    row: Tile row.
    col: Tile column.
    Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
    """
#     tissue_percent =tissue_percent(np_tile)
    color_factor = hsv_purple_pink_factor(np_tile)
    s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
    amount = tissue_quantity(tissue_percent)
    quantity_factor = tissue_quantity_factor(amount)
    combined_factor = color_factor * s_and_v_factor * quantity_factor
    score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
    return score, color_factor, s_and_v_factor, quantity_factor


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).
    Args:
    np_img: Image as a NumPy array.
    Returns:
    The percentage of the NumPy array that is tissue.
    """
    return 100 - mask_percent(np_img)

class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed




class TissueQuantity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def tissue_quantity_factor(amount):
    """
    Obtain a scoring factor based on the quantity of tissue in a tile.
    Args:
    amount: Tissue amount as a TissueQuantity enum value.
    Returns:
    Scoring factor based on the tile tissue quantity.
    """
    if amount == TissueQuantity.HIGH:
        quantity_factor = 1.0
    elif amount == TissueQuantity.MEDIUM:
        quantity_factor = 0.2
    elif amount == TissueQuantity.LOW:
        quantity_factor = 0.1
    else:
        quantity_factor = 0.0
    return quantity_factor


def tissue_quantity(tissue_percentage):
    """
    Obtain TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE) for corresponding tissue percentage.
    Args:
    tissue_percentage: The tile tissue percentage.
    Returns:
    TissueQuantity enum member (HIGH, MEDIUM, LOW, or NONE).
    """
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        return TissueQuantity.HIGH
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        return TissueQuantity.MEDIUM
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        return TissueQuantity.LOW
    else:
        return TissueQuantity.NONE



    