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

