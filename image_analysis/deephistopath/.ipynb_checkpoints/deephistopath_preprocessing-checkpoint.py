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
from PIL import Image

import openslide
from openslide import OpenSlide, OpenSlideUnsupportedFormatError,OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage
from itertools import chain
import matplotlib.pylab as plt
import datetime


ADDITIONAL_NP_STATS = False

def open_slide(filepath=None):
    slide = openslide.OpenSlide(filepath)
    
    return slide


def show_slide(slide=None):
    """
    Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.
    Args:
    slide_number: The slide number.
    """
    pil_img = slide_to_scaled_pil_image(slide)[0]
    pil_img.show()


def create_tile_generator(slide=None, tile_size=None, overlap=None):
    """
      Create a tile generator for the given slide.
      This generator is able to extract tiles from the overall
      whole-slide image.
      Args:
        slide: An OpenSlide object representing a whole-slide image.
        tile_size: The width and height of a square tile to be generated.
        overlap: Number of pixels by which to overlap the tiles.
      Returns:
        A DeepZoomGenerator object representing the tile generator. Each
        extracted tile is a PIL Image with shape
        (tile_size, tile_size, channels).
        Note: This generator is not a true "Python generator function", but
        rather is an object that is capable of extracting individual tiles.
    """
    generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
      
    return generator

def get_20x_zoom_level(slide=None, generator=None):
    """
      Return the zoom level that corresponds to a 20x magnification.
      The generator can extract tiles from multiple zoom levels,
      downsampling by a factor of 2 per level from highest to lowest
      resolution.
      Args:
        slide: An OpenSlide object representing a whole-slide image.
        generator: A DeepZoomGenerator object representing a tile generator.
          Note: This generator is not a true "Python generator function",
          but rather is an object that is capable of extracting individual
          tiles.
      Returns:
        Zoom level corresponding to a 20x magnification, or as close as
        possible.
      """
    highest_zoom_level = generator.level_count - 1  # 0-based indexing
    try:
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        # `mag / 20` gives the downsampling factor between the slide's
        # magnification and the desired 20x magnification.
        # `(mag / 20) / 2` gives the zoom level offset from the highest
        # resolution level, based on a 2x downsampling factor in the
        # generator.
        offset = math.floor((mag / 20) / 2) #20 -> 40
        level = highest_zoom_level - offset
    except (ValueError, KeyError) as e:
        # In case the slide magnification level is unknown, just
        # use the highest resolution.
        level = highest_zoom_level
    return level

def process_slide(filepath=None, tile_size=None, overlap=None):
    """
      Generate all possible tile indices for a whole-slide image.
      Given a slide number, tile size, and overlap, generate
      all possible (slide_num, tile_size, overlap, zoom_level, col, row)
      indices.
      Args:
        slide_num: Slide image number as an integer.
        folder: Directory in which the slides folder is stored, as a string.
          This should contain either a `training_image_data` folder with
          images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
          folder with images in the format `TUPAC-TE-###.svs`.
        training: Boolean for training or testing datasets.
        tile_size: The width and height of a square tile to be generated.
        overlap: Number of pixels by which to overlap the tiles.
      Returns:
        A list of (slide_num, tile_size, overlap, zoom_level, col, row)
        integer index tuples representing possible tiles to extract.
      """
    # Open slide.
    slide = open_slide(filepath)
    # Create tile generator.
    generator = create_tile_generator(slide, tile_size, overlap)
    # Get 20x zoom level.
    zoom_level = get_20x_zoom_level(slide, generator)
    # Generate all possible (zoom_level, col, row) tile index tuples.
    cols, rows = generator.level_tiles[zoom_level]
    tile_indices = [(tile_size, overlap, zoom_level, col, row)
                  for col in range(cols) for row in range(rows)]
    return tile_indices

def process_tile_index(tile_index, filepath):
    """
    Generate a tile from a tile index.
    Given a (slide_num, tile_size, overlap, zoom_level, col, row) tile
    index, generate a (slide_num, tile) tuple.
    Args:
    tile_index: A (slide_num, tile_size, overlap, zoom_level, col, row)
      integer index tuple representing a tile to extract.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.
    Returns:
    A (slide_num, tile) tuple, where slide_num is an integer, and tile
    is a 3D NumPy array of shape (tile_size, tile_size, channels) in
    RGB format.
    """
    tile_size, overlap, zoom_level, col, row = tile_index
    # Open slide.
    slide = open_slide(filename)
    # Create tile generator.
    generator = create_tile_generator(slide, tile_size, overlap)
    # Generate tile.
    tile = np.asarray(generator.get_tile(zoom_level, (col, row)))
    return (tile)


# Filter Tile For Dimensions & Tissue Threshold

def optical_density(tile=None):
    """
    Convert a tile to optical density values.
    Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
    Returns:
    A 3D NumPy array of shape (tile_size, tile_size, channels)
    representing optical density values.
    """
    tile = tile.astype(np.float64)
    #od = -np.log10(tile/255 + 1e-8)
    od = -np.log((tile+1)/240)
    return od



# def get_tissue_mask(
#         thumbnail_im,
#         deconvolve_first=False, stain_unmixing_routine_kwargs={},
#         n_thresholding_steps=1, sigma=0., min_size=500):
#     """Get binary tissue mask from slide thumbnail.
#     Parameters
#     -----------
#     thumbnail_im : np array
#         (m, n, 3) nd array of thumbnail RGB image
#         or (m, n) nd array of thumbnail grayscale image
#     deconvolve_first : bool
#         use hematoxylin channel to find cellular areas?
#         This will make things ever-so-slightly slower but is better in
#         getting rid of sharpie marker (if it's green, for example).
#         Sometimes things work better without it, though.
#     stain_matrix_method : str
#         see deconv_color method in seed_utils
#     n_thresholding_steps : int
#         number of gaussian smoothign steps
#     sigma : float
#         sigma of gaussian filter
#     min_size : int
#         minimum size (in pixels) of contiguous tissue regions to keep
#     Returns
#     --------
#     np bool array
#         largest contiguous tissue region.
#     np int32 array
#         each unique value represents a unique tissue region
#     """
#     if deconvolve_first and (len(thumbnail_im.shape) == 3):
#         # deconvolvve to ge hematoxylin channel (cellular areas)
#         # hematoxylin channel return shows MINIMA so we invert
#         stain_unmixing_routine_kwargs['stains'] = ['hematoxylin', 'eosin']
#         Stains, _, _ = color_deconvolution_routine(
#             thumbnail_im, **stain_unmixing_routine_kwargs)
#         thumbnail = 255 - Stains[..., 0]

#     elif len(thumbnail_im.shape) == 3:
#         # grayscale thumbnail (inverted)
#         thumbnail = 255 - cv2.cvtColor(thumbnail_im, cv2.COLOR_BGR2GRAY)

#     else:
#         thumbnail = thumbnail_im

#     for _ in range(n_thresholding_steps):

#         # gaussian smoothing of grayscale thumbnail
#         if sigma > 0.0:
#             thumbnail = gaussian(
#                 thumbnail, sigma=sigma,
#                 output=None, mode='nearest', preserve_range=True)

#         # get threshold to keep analysis region
#         try:
#             thresh = threshold_otsu(thumbnail[thumbnail > 0])
#         except ValueError:  # all values are zero
#             thresh = 0

#         # replace pixels outside analysis region with upper quantile pixels
#         thumbnail[thumbnail < thresh] = 0

#     # convert to binary
#     mask = 0 + (thumbnail > 0)

#     # find connected components
#     labeled, _ = ndimage.label(mask)

#     # only keep
#     unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
#     discard = np.in1d(labeled, unique[counts < min_size])
#     discard = discard.reshape(labeled.shape)
#     labeled[discard] = 0

#     # largest tissue region
#     mask = labeled == unique[np.argmax(counts)]

#     return labeled, mask

# def _deconv_color(im, **kwargs):
#     """Wrap around color_deconvolution_routine (compatibility)."""
#     Stains, _, _ = color_deconvolution_routine(im, **kwargs)
#     return Stains, 0

# def get_slide_thumbnail(gc, slide_id):
#     """Get slide thumbnail using girder client.
#     Parameters
#     -------------
#     gc : object
#         girder client to use
#     slide_id : str
#         girder ID of slide
#     Returns
#     ---------
#     np array
#         RGB slide thumbnail at lowest level
#     """
#     getStr = "/item/%s/tiles/thumbnail" % (slide_id)
#     resp = gc.get(getStr, jsonResp=False)
#     return get_image_from_htk_response(resp)




# def np_info(np_arr, name=None, elapsed=None):
#     """
#     Display information (shape, type, max, min, etc) about a NumPy array.
#     Args:
#     np_arr: The NumPy array.
#     name: The (optional) name of the array.
#     elapsed: The (optional) time elapsed to perform a filtering operation.
#     """

#     if name is None:
#         name = "NumPy Array"
#     if elapsed is None:
#         elapsed = "---"

#     if ADDITIONAL_NP_STATS is False:
#         print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
#     else:
#         # np_arr = np.asarray(np_arr)
#         max = np_arr.max()
#         min = np_arr.min()
#         mean = np_arr.mean()
#         is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
#         print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
#           name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))
    
# def mask_rgb(rgb, mask):
#     """
#     Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
#     Args:
#     rgb: RGB image as a NumPy array.
#     mask: An image mask to determine which pixels in the original image should be displayed.
#     Returns:
#     NumPy array representing an RGB image with mask applied.
#     """
#     t = Time()
#     result = rgb * np.dstack([mask, mask, mask])
#     np_info(result, "Mask RGB", t.elapsed())
#     return result

# def mask_percent(np_img):
#     """
#     Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
#     Args:
#     np_img: Image as a NumPy array.
#     Returns:
#     The percentage of the NumPy array that is masked.
#     """
#     if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
#         np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
#         mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
#     else:
#         mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
#     return mask_percentage


# def tissue_percent(np_img):
#     """
#     Determine the percentage of a NumPy array that is tissue (not masked).
#     Args:
#     np_img: Image as a NumPy array.
#     Returns:
#     The percentage of the NumPy array that is tissue.
#     """
#     return 100 - mask_percent(np_img)

# class Time:
#     """
#     Class for displaying elapsed time.
#     """

#     def __init__(self):
#         self.start = datetime.datetime.now()

#     def elapsed_display(self):
#         time_elapsed = self.elapsed()
#         print("Time elapsed: " + str(time_elapsed))

#     def elapsed(self):
#         self.end = datetime.datetime.now()
#         time_elapsed = self.end - self.start
#         return time_elapsed
