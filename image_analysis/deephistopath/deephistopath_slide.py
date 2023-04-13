import glob
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import openslide
from openslide import OpenSlide, OpenSlideUnsupportedFormatError,OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

from openslide import OpenSlideError
import os
import PIL
from PIL import Image,ImageDraw
import re
import sys
import os
from os.path import basename
import pathlib
from pathlib import Path
import shutil  
from enum import Enum

# print(os.getcwd())
from deephistopath_utils import Time
# from deephistopath_slide import *
from deephistopath_preprocessing import *
from deephistopath_filters import *

fig = plt.figure(figsize =(10,10))

SCALE_FACTOR = 32
THUMBNAIL_SIZE = 300


TISSUE_HIGH_THRESH = 80
TISSUE_LOW_THRESH = 10

ROW_TILE_SIZE = 1024
COL_TILE_SIZE = 1024
NUM_TOP_TILES = 50

DISPLAY_TILE_SUMMARY_LABELS = False
TILE_LABEL_TEXT_SIZE = 10
LABEL_ALL_TILES_IN_TOP_TILE_SUMMARY = False
BORDER_ALL_TILES_IN_TOP_TILE_SUMMARY = False

TILE_BORDER_SIZE = 2  # The size of the colored rectangular border around summary tiles.

HIGH_COLOR = (0, 255, 0)
MEDIUM_COLOR = (255, 255, 0)
LOW_COLOR = (255, 165, 0)
NONE_COLOR = (255, 0, 0)

FADED_THRESH_COLOR = (128, 255, 128)
FADED_MEDIUM_COLOR = (255, 255, 128)
FADED_LOW_COLOR = (255, 210, 128)
FADED_NONE_COLOR = (255, 128, 128)

SUMMARY_TITLE_TEXT_COLOR = (0, 0, 0)
SUMMARY_TITLE_TEXT_SIZE = 24
SUMMARY_TILE_TEXT_COLOR = (255, 255, 255)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_TEXT_SIZE = 36
TILE_TEXT_BACKGROUND_COLOR = (255, 255, 255)
TILE_TEXT_W_BORDER = 5
TILE_TEXT_H_BORDER = 4

HSV_PURPLE = 270
HSV_PINK = 330

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



def slide_stats(img_paths=None,img_names=None):
    """
    Display statistics/graphs about slides.
    Give all image paths and image filenames 
    """
    t = Time()

    if not os.path.exists(STATS_DIR):
        os.mkdir(STATS_DIR)

    num_images = len(img_paths)
    slide_stats = []
    for slide_num in range(0, num_images):
        slide_name = img_names[slide_num]
        slide_filepath = img_paths[slide_num]
        print("Opening Slide #%d: %s" % (slide_num,slide_name ))
        slide = open_slide(slide_filepath)
        (width, height) = slide.dimensions
        print("  Dimensions: {:,d} x {:,d}".format(width, height))
        slide_stats.append((width, height))
    
    max_width = 0
    max_height = 0
    min_width = sys.maxsize
    min_height = sys.maxsize
    total_width = 0
    total_height = 0
    total_size = 0
    which_max_width = 0
    which_max_height = 0
    which_min_width = 0
    which_min_height = 0
    max_size = 0
    min_size = sys.maxsize
    which_max_size = 0
    which_min_size = 0
    for z in range(0, num_images):
        (width, height) = slide_stats[z]
        if width > max_width:
            max_width = width
            which_max_width = z + 1
        if width < min_width:
            min_width = width
            which_min_width = z + 1
        if height > max_height:
            max_height = height
            which_max_height = z + 1
        if height < min_height:
            min_height = height
            which_min_height = z + 1
        size = width * height
        if size > max_size:
            max_size = size
            which_max_size = z + 1
        if size < min_size:
            min_size = size
            which_min_size = z + 1
        total_width = total_width + width
        total_height = total_height + height
        total_size = total_size + size
    

    avg_width = total_width / num_images
    avg_height = total_height / num_images
    avg_size = total_size / num_images

    stats_string = ""
    stats_string += "%-11s {:14,d} pixels (slide #%d)".format(max_width) % ("Max width:", which_max_width)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_height) % ("Max height:", which_max_height)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(max_size) % ("Max size:", which_max_size)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_width) % ("Min width:", which_min_width)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_height) % ("Min height:", which_min_height)
    stats_string += "\n%-11s {:14,d} pixels (slide #%d)".format(min_size) % ("Min size:", which_min_size)
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_width)) % "Avg width:"
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_height)) % "Avg height:"
    stats_string += "\n%-11s {:14,d} pixels".format(round(avg_size)) % "Avg size:"
    stats_string += "\n"
    print(stats_string)


    stats_string += "\nslide number,width,height"
    for i in range(0, len(slide_stats)):
        (width, height) = slide_stats[i]
        stats_string += "\n%d,%d,%d" % (i + 1, width, height)
    stats_string += "\n"


    stats_file = open(os.path.join(STATS_DIR, "stats.txt"), "a")
    stats_file.write(stats_string)
    stats_file.close()

    t.elapsed_display()


    x, y = zip(*slide_stats)
    colors = np.random.rand(num_images)
    sizes = [10 for n in range(num_images)]
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.title("SVS Image Sizes")
    plt.set_cmap("prism")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "svs-image-sizes.png"))
    plt.show()
    

    plt.clf()
    plt.scatter(x, y, s=sizes, c=colors, alpha=0.7)
    plt.xlabel("width (pixels)")
    plt.ylabel("height (pixels)")
    plt.title("SVS Image Sizes (Labeled with slide numbers)")
    plt.set_cmap("prism")
    for i in range(num_train_images):
        snum = i + 1
        plt.annotate(str(snum), (x[i], y[i]))
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "svs-image-sizes-slide-numbers.png"))
    plt.show()

    plt.clf()
    area = [w * h / 1000000 for (w, h) in slide_stats]
    plt.hist(area, bins=64)
    plt.xlabel("width x height (M of pixels)")
    plt.ylabel("# images")
    plt.title("Distribution of image sizes in millions of pixels")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "distribution-of-svs-image-sizes.png"))
    plt.show()

    plt.clf()
    whratio = [w / h for (w, h) in slide_stats]
    plt.hist(whratio, bins=64)
    plt.xlabel("width to height ratio")
    plt.ylabel("# images")
    plt.title("Image shapes (width to height)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "w-to-h.png"))
    plt.show()                          

    plt.clf()
    hwratio = [h / w for (w, h) in slide_stats]
    plt.hist(hwratio, bins=64)
    plt.xlabel("height to width ratio")
    plt.ylabel("# images")
    plt.title("Image shapes (height to width)")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "h-to-w.png"))
    plt.show()
                             
                  
                             
def save_thumbnail(pil_img, size, path, display_path=False):
    """
    Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.
    Args:
    pil_img: The PIL image to save as a thumbnail.
    size:  The maximum width or height of the thumbnail.
    path: The path to the thumbnail.
    display_path: If True, display thumbnail path in console.
    """
    max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
    img = pil_img.resize(max_size, PIL.Image.BILINEAR)
    if display_path:
        print("Saving thumbnail to: " + path)
#     dir = os.path.dirname(path)
#     if dir != '' and not os.path.exists(dir):
#         os.makedirs(dir)
    img.save(path)


def summary_and_tiles(img_path = None, filtered_img =None, display=True, save_summary=False, save_data=False, save_top_tiles=False):
    """
    Generate tile summary and top tiles for slide.
    Args:
    slide_num: The slide number.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    save_data: If True, save tile data to csv file.
    save_top_tiles: If True, save top tiles to files.
    """
#     img_path = slide.get_filter_image_result(slide_num)
#     filtered_img = slide.open_image_np(img_path)

    tile_sum = score_tiles(img_path,filtered_img)
    if save_data:
        save_tile_data(tile_sum)
    generate_tile_summaries(tile_sum, np_img = filtered_img,img_path=img_path, display=display, save_summary=save_summary)
#     generate_top_tile_summaries(tile_sum, np_img = filtered_image, display=display, save_summary=save_summary)
    if save_top_tiles:
        for tile in tile_sum.top_tiles():
              tile.save_tile()
    return tile_sum


def score_tiles(img_path, filtered_img, dimensions=None, small_tile_in_tile=False):
    """
    Score all tiles for a slide and return the results in a TileSummary object.
    Args:
    slide_num: The slide number.
    np_img: Optional image as a NumPy array.
    dimensions: Optional tuple consisting of (original width, original height, new width, new height). Used for dynamic
      tile retrieval.
    small_tile_in_tile: If True, include the small NumPy image in the Tile objects.
    Returns:
    TileSummary object which includes a list of Tile objects containing information about each tile.
    """
    if dimensions is None:
#         img_path = slide.get_filter_image_result(slide_num)
       img, o_w, o_h, w, h = slide_to_scaled_pil_image(img_path)#slide.parse_dimensions_from_image_filename(img_path)
    else:
        o_w, o_h, w, h = dimensions

    if filtered_img is None:
        filtered_img = img #slide.open_image_np(img_path)

    row_tile_size = round(ROW_TILE_SIZE / SCALE_FACTOR)  # use round?
    col_tile_size = round(COL_TILE_SIZE / SCALE_FACTOR)  # use round?

    num_row_tiles, num_col_tiles = get_num_tiles(h, w, row_tile_size, col_tile_size)

    tile_sum = TileSummary(img_path = img_path,
                           filtered_img = filtered_img,
                         orig_w=o_w,
                         orig_h=o_h,
                         orig_tile_w=COL_TILE_SIZE,
                         orig_tile_h=ROW_TILE_SIZE,
                         scaled_w=w,
                         scaled_h=h,
                         scaled_tile_w=col_tile_size,
                         scaled_tile_h=row_tile_size,
                         tissue_percentage=tissue_percent(filtered_img),
                         num_col_tiles=num_col_tiles,
                         num_row_tiles=num_row_tiles)

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0
    tile_indices = get_tile_indices(h, w, row_tile_size, col_tile_size)
    print("Len of tile_indices",len(tile_indices))
    for t in tile_indices:
        count += 1  # tile_num
        r_s,r_e, c_s, c_e, r, c = t # 
        np_tile = filtered_img[r_s:r_e, c_s:c_e]
        t_p = tissue_percent(np_tile)
        amount = tissue_quantity(t_p)
        if amount == TissueQuantity.HIGH:
             high += 1
        elif amount == TissueQuantity.MEDIUM:
             medium += 1
        elif amount == TissueQuantity.LOW:
             low += 1
        elif amount == TissueQuantity.NONE:
             none += 1
        o_c_s, o_r_s = small_to_large_mapping((c_s, r_s), (o_w, o_h))
        o_c_e, o_r_e = small_to_large_mapping((c_e, r_e), (o_w, o_h))

        # pixel adjustment in case tile dimension too large (for example, 1025 instead of 1024)
        if (o_c_e - o_c_s) > COL_TILE_SIZE:
             o_c_e -= 1
        if (o_r_e - o_r_s) > ROW_TILE_SIZE:
             o_r_e -= 1

        score, color_factor, s_and_v_factor, quantity_factor = score_tile(np_tile, t_p, r, c)

        np_scaled_tile = np_tile if small_tile_in_tile else None
        tile = Tile(tile_sum, img_path,np_scaled_tile, count, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
                    o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score)
        tile_sum.tiles.append(tile)

    tile_sum.count = count
    tile_sum.high = high
    tile_sum.medium = medium
    tile_sum.low = low
    tile_sum.none = none

    tiles_by_score = tile_sum.tiles_by_score()
    rank = 0
    for t in tiles_by_score:
        rank += 1
    t.rank = rank

    return tile_sum

def get_num_tiles(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain the number of vertical and horizontal tiles that an image can be divided into given a row tile size and
    a column tile size.
    Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.
    Returns:
    Tuple consisting of the number of vertical tiles and the number of horizontal tiles that the image can be divided
    into given the row tile size and the column tile size.
    """
    num_row_tiles = math.ceil(rows / row_tile_size)
    num_col_tiles = math.ceil(cols / col_tile_size)
    return num_row_tiles, num_col_tiles

def score_tile(np_tile, tissue_percent, row, col):
    """
    Score tile based on tissue percentage, color factor, saturation/value factor, and tissue quantity factor.
    Args:
    np_tile: Tile as NumPy array.
    tissue_percent: The percentage of the tile judged to be tissue.
    slide_num: Slide number.

    col: Tile column.
    Returns tuple consisting of score, color factor, saturation/value factor, and tissue quantity factor.
    """
    color_factor = hsv_purple_pink_factor(np_tile)
    s_and_v_factor = hsv_saturation_and_value_factor(np_tile)
    amount = tissue_quantity(tissue_percent)
    quantity_factor = tissue_quantity_factor(amount)
    combined_factor = color_factor * s_and_v_factor * quantity_factor
    score = (tissue_percent ** 2) * np.log(1 + combined_factor) / 1000.0
    # scale score to between 0 and 1
    score = 1.0 - (10.0 / (10.0 + score))
    return score, color_factor, s_and_v_factor, quantity_factor

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


class TileSummary:
    """
    Class for tile summary information.
    """

#     slide_num = None
    orig_w = None
    orig_h = None
    orig_tile_w = None
    orig_tile_h = None
    scale_factor = SCALE_FACTOR
    scaled_w = None
    scaled_h = None
    scaled_tile_w = None
    scaled_tile_h = None
    mask_percentage = None
    num_row_tiles = None
    num_col_tiles = None

    count = 0
    high = 0
    medium = 0
    low = 0
    none = 0

    def __init__(self, img_path, filtered_img,orig_w, orig_h, orig_tile_w, orig_tile_h, scaled_w, scaled_h, scaled_tile_w,
               scaled_tile_h, tissue_percentage, num_col_tiles, num_row_tiles): #slide_num
        self.img_path = img_path
        self.filtered_img = filtered_img
        self.slide_num = 1
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_tile_w = orig_tile_w
        self.orig_tile_h = orig_tile_h
        self.scaled_w = scaled_w
        self.scaled_h = scaled_h
        self.scaled_tile_w = scaled_tile_w
        self.scaled_tile_h = scaled_tile_h
        self.tissue_percentage = tissue_percentage
        self.num_col_tiles = num_col_tiles
        self.num_row_tiles = num_row_tiles
        self.tiles = []

    def __str__(self):
        return summary_title(self) + "\n" + summary_stats(self)

    def mask_percentage(self):
        """
        Obtain the percentage of the slide that is masked.
        Returns:
           The amount of the slide that is masked as a percentage.
        """
        return 100 - self.tissue_percentage

    def num_tiles(self):
        """
        Retrieve the total number of tiles.
        Returns:
          The total number of tiles (number of rows * number of columns).
        """
        return self.num_row_tiles * self.num_col_tiles

    def tiles_by_tissue_percentage(self):
        """
        Retrieve the tiles ranked by tissue percentage.
        Returns:
           List of the tiles ranked by tissue percentage.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.tissue_percentage, reverse=True)
        return sorted_list

    def tiles_by_score(self):
        """
        Retrieve the tiles ranked by score.
        Returns:
           List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list

    def top_tiles(self):
        """
        Retrieve the top-scoring tiles.
        Returns:
           List of the top-scoring tiles.
        """
        sorted_tiles = self.tiles_by_score()
        top_tiles = sorted_tiles[:NUM_TOP_TILES]
        return top_tiles

    def get_tile(self, row, col):
        """
        Retrieve tile by row and column.
        Args:
          row: The row
          col: The column
        Returns:
          Corresponding Tile object.
        """
        tile_index = (row - 1) * self.num_col_tiles + (col - 1)
        tile = self.tiles[tile_index]
        return tile

    def display_summaries(self):
        """
        Display summary images.
        """
        summary_and_tiles(self.img_path, display=True, save_summary=False, save_data=False, save_top_tiles=False)


def small_to_large_mapping(small_pixel, large_dimensions):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
    small_pixel: The scaled-down width and height.
    large_dimensions: The width and height of the original whole-slide image.
    Returns:
    Tuple consisting of the scaled-up width and height.
    """
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
    large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
    return large_x, large_y


def summary_stats(tile_summary):
    """
    Obtain various stats about the slide tiles.
    Args:
    tile_summary: TileSummary object.
    Returns:
     Various stats about the slide tiles as a string.
    """
    return "Original Dimensions: %dx%d\n" % (tile_summary.orig_w, tile_summary.orig_h) + \
         "Original Tile Size: %dx%d\n" % (tile_summary.orig_tile_w, tile_summary.orig_tile_h) + \
         "Scale Factor: 1/%dx\n" % tile_summary.scale_factor + \
         "Scaled Dimensions: %dx%d\n" % (tile_summary.scaled_w, tile_summary.scaled_h) + \
         "Scaled Tile Size: %dx%d\n" % (tile_summary.scaled_tile_w, tile_summary.scaled_tile_w) + \
         "Total Mask: %3.2f%%, Total Tissue: %3.2f%%\n" % (
           tile_summary.mask_percentage(), tile_summary.tissue_percentage) + \
         "Tiles: %dx%d = %d\n" % (tile_summary.num_col_tiles, tile_summary.num_row_tiles, tile_summary.count) + \
         " %5d (%5.2f%%) tiles >=%d%% tissue\n" % (
           tile_summary.high, tile_summary.high / tile_summary.count * 100, TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >=%d%% and <%d%% tissue\n" % (
           tile_summary.medium, tile_summary.medium / tile_summary.count * 100, TISSUE_LOW_THRESH,
           TISSUE_HIGH_THRESH) + \
         " %5d (%5.2f%%) tiles >0%% and <%d%% tissue\n" % (
           tile_summary.low, tile_summary.low / tile_summary.count * 100, TISSUE_LOW_THRESH) + \
         " %5d (%5.2f%%) tiles =0%% tissue" % (tile_summary.none, tile_summary.none / tile_summary.count * 100)


def get_tile_indices(rows, cols, row_tile_size, col_tile_size):
    """
    Obtain a list of tile coordinates (starting row, ending row, starting column, ending column, row number, column number).
    Args:
    rows: Number of rows.
    cols: Number of columns.
    row_tile_size: Number of pixels in a tile row.
    col_tile_size: Number of pixels in a tile column.
    Returns:
    List of tuples representing tile coordinates consisting of starting row, ending row,
    starting column, ending column, row number, column number.
    """
    indices = list()
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    for r in range(0, num_row_tiles):
        start_r = r * row_tile_size
        end_r = ((r + 1) * row_tile_size) if (r < num_row_tiles - 1) else rows
    for c in range(0, num_col_tiles):
        start_c = c * col_tile_size
        end_c = ((c + 1) * col_tile_size) if (c < num_col_tiles - 1) else cols
        indices.append((start_r, end_r, start_c, end_c, r + 1, c + 1))
    return indices


class Tile:
    """
    Class for information about a tile.
    """

    def __init__(self, tile_summary,img_path, np_scaled_tile, tile_num, r, c, r_s, r_e, c_s, c_e, o_r_s, o_r_e, o_c_s,
               o_c_e, t_p, color_factor, s_and_v_factor, quantity_factor, score):
        self.tile_summary = tile_summary
        self.img_path = img_path
#         self.slide_num = slide_num
        self.np_scaled_tile = np_scaled_tile
        self.tile_num = tile_num
        self.r = r
        self.c = c
        self.r_s = r_s
        self.r_e = r_e
        
        self.c_s = c_s
        self.c_e = c_e
        self.o_r_s = o_r_s
        self.o_r_e = o_r_e
        self.o_c_s = o_c_s
        self.o_c_e = o_c_e
        self.tissue_percentage = t_p
        self.color_factor = color_factor
        self.s_and_v_factor = s_and_v_factor
        self.quantity_factor = quantity_factor
        self.score = score

    def __str__(self):
        return "[Tile #%d, Row #%d, Column #%d, Tissue %4.2f%%, Score %0.4f]" % (
      self.tile_num, self.r, self.c, self.tissue_percentage, self.score)

    def __repr__(self):
        return "\n" + self.__str__()

    def mask_percentage(self):
        return 100 - self.tissue_percentage

    def tissue_quantity(self):
        return tissue_quantity(self.tissue_percentage)

    def get_pil_tile(self):
        return tile_to_pil_tile(self)

    def get_np_tile(self):
        return tile_to_np_tile(self)

    def save_tile(self):
        save_display_tile(self, save=True, display=False)

    def display_tile(self):
        save_display_tile(self, save=False, display=True)

    def display_with_histograms(self):
        display_tile(self, rgb_histograms=True, hsv_histograms=True)

    def get_np_scaled_tile(self):
        return self.np_scaled_tile
    
    def get_pil_scaled_tile(self):
        return np_to_pil(self.np_scaled_tile)

    
    def tiles_by_score(self):
        """
        Retrieve the tiles ranked by score.
        Returns:
           List of the tiles ranked by score.
        """
        sorted_list = sorted(self.tiles, key=lambda t: t.score, reverse=True)
        return sorted_list


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
    """
    Create a PIL summary image including top title area and right side and bottom padding.
    Args:
    np_img: Image as a NumPy array.
    title_area_height: Height of the title area at the top of the summary image.
    row_tile_size: The tile size in rows.
    col_tile_size: The tile size in columns.
    num_row_tiles: The number of row tiles.
    num_col_tiles: The number of column tiles.
    Returns:
    Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
    potentially a top title area and right side and bottom padding.
    """
#     np_img= pil_to_np_rgb(np_img)
    r = row_tile_size * num_row_tiles + title_area_height
    c = col_tile_size * num_col_tiles
    summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
    # add gray edges so that tile text does not get cut off
    summary_img.fill(120)
    # color title area white
    summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
    summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
    summary = np_to_pil(summary_img)
    return summary


def generate_tile_summaries(tile_sum, np_img=None, img_path= None, display=True, save_summary=False):
    """
    Generate summary images/thumbnails showing a 'heatmap' representation of the tissue segmentation of all tiles.
    Args:
    tile_sum: TileSummary object.
    np_img: Image as a NumPy array.
    display: If True, display tile summary to screen.
    save_summary: If True, save tile summary images.
    """
    z = 300  # height of area at top of summary slide
#     slide_num = tile_sum.slide_num
    rows = tile_sum.scaled_h
    cols = tile_sum.scaled_w
    row_tile_size = tile_sum.scaled_tile_h
    col_tile_size = tile_sum.scaled_tile_w
    num_row_tiles, num_col_tiles = get_num_tiles(rows, cols, row_tile_size, col_tile_size)
    summary = create_summary_pil_img(np_img, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw = ImageDraw.Draw(summary)

#     original_img_path = slide.get_training_image_path(slide_num)
#     np_orig = slide.open_image_np(original_img_path)
    np_orig, o_w, o_h, w, h = slide_to_scaled_pil_image(img_path)
    np_orig = pil_to_np_rgb(np_orig) #should be in rgb format 
    summary_orig = create_summary_pil_img(np_orig, z, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles)
    draw_orig = ImageDraw.Draw(summary_orig)

    for t in tile_sum.tiles:
        border_color = tile_border_color(t.tissue_percentage)
        tile_border(draw, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)
        tile_border(draw_orig, t.r_s + z, t.r_e + z, t.c_s, t.c_e, border_color)

    summary_txt = summary_title(tile_sum) + "\n" + summary_stats(tile_sum)

#     summary_font = ImageFont.truetype(SUMMARY_TITLE_FONT_PATH, size=SUMMARY_TITLE_TEXT_SIZE)
    draw.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR )#, font=summary_font)
    draw_orig.text((5, 5), summary_txt, SUMMARY_TITLE_TEXT_COLOR )#, font=summary_font)

    if DISPLAY_TILE_SUMMARY_LABELS:
        count = 0
        for t in tile_sum.tiles:
            count += 1
            label = "R%d\nC%d" % (t.r, t.c)
#             font = ImageFont.truetype(FONT_PATH, size=TILE_LABEL_TEXT_SIZE)
            # drop shadow behind text
            draw.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0))#, font=font)
            draw_orig.text(((t.c_s + 3), (t.r_s + 3 + z)), label, (0, 0, 0) )#, font=font)

            draw.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR )#, font=font)
            draw_orig.text(((t.c_s + 2), (t.r_s + 2 + z)), label, SUMMARY_TILE_TEXT_COLOR)#, font=font)

    if display:
        plt.imshow(summary_orig)
        plt.imshow(summary)
        plt.show()
#         summary.show()
#         summary_orig.show()
    if save_summary:
        save_tile_summary_image(summary, slide_num)
        save_tile_summary_on_original_image(summary_orig, slide_num)

def summary_title(tile_summary):
    """
    Obtain tile summary title.
    Args:
    tile_summary: TileSummary object.
    Returns:
     The tile summary title.
    """
    return "Slide %03d Tile Summary:" % tile_summary.slide_num

def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=TILE_BORDER_SIZE):
    """
    Draw a border around a tile with width TILE_BORDER_SIZE.
    Args:
    draw: Draw object for drawing on PIL image.
    r_s: Row starting pixel.
    r_e: Row ending pixel.
    c_s: Column starting pixel.
    c_e: Column ending pixel.
    color: Color of the border.
    border_size: Width of tile border in pixels.
    """
    for x in range(0, border_size):
        draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)


def tile_border_color(tissue_percentage):
    """
    Obtain the corresponding tile border color for a particular tile tissue percentage.
    Args:
    tissue_percentage: The tile tissue percentage
    Returns:
    The tile border color corresponding to the tile tissue percentage.
    """
    if tissue_percentage >= TISSUE_HIGH_THRESH:
        border_color = HIGH_COLOR
    elif (tissue_percentage >= TISSUE_LOW_THRESH) and (tissue_percentage < TISSUE_HIGH_THRESH):
        border_color = MEDIUM_COLOR
    elif (tissue_percentage > 0) and (tissue_percentage < TISSUE_LOW_THRESH):
        border_color = LOW_COLOR
    else:
        border_color = NONE_COLOR
    return border_color

def tile_to_np_tile(tile):
    """
    Convert tile information into the corresponding tile as a NumPy image read from the whole-slide image file.
    Args:
    tile: Tile object.
    Return:
    Tile as a NumPy image.
    """
    pil_img = tile_to_pil_tile(tile)
    np_img = pil_to_np_rgb(pil_img)
    return np_img

def tile_to_pil_tile(tile):
    """
    Convert tile information into the corresponding tile as a PIL image read from the whole-slide image file.
    Args:
    tile: Tile object.
    Return:
    Tile as a PIL image.
    """
    t = tile
    slide_filepath = t.img_path# get_training_slide_path(t.slide_num)
    s = open_slide(slide_filepath)

    x, y = t.o_c_s, t.o_r_s
    w, h = t.o_c_e - t.o_c_s, t.o_r_e - t.o_r_s
    tile_region = s.read_region((x, y), 0, (w, h))
    # RGBA to RGB
    pil_img = tile_region.convert("RGB")
    return pil_img

