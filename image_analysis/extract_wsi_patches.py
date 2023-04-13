"""
## Automated WSI patch extraction 
The script focuses on extraction of valid patches from TCGA- GBM and LGG whole slide imgaes.A series of pre-processing steps are applied for selection of top patches per given image slide.

NOTE: Make sure the packages are installed in the virtual environment before running the script.
"""
import os,sys
sys.path.insert(1, 'deephistopath/')

import argparse
from pathlib import Path
import os
from itertools import chain
import numpy as np
import pandas as pd
from PIL import Image
import math
from openslide import OpenSlide, OpenSlideUnsupportedFormatError,OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

import skimage.io
import skimage.measure
import skimage.color
from skimage import  img_as_ubyte
 
from wsi_utils  import open_slide, read_wsi, score_tile,construct_colored_wsi, mask_rgb, get_contours, construct_bags,tissue_percent,estimate_blur,small_to_large_mapping
import deephistopath_filters as filters
import deephistopath_utils as utils


SCALE_FACTOR = 16
PATCH_SIZE = 256
LOW_RES = False
BASE_PATH = "path/to/working/dir"
SAVE=True

def get_files(file_path):
    """
    Extract all the WSI-Whole Slide Images as list 
    :param file_path: dir/path of whole slide images with extension .svs
    """
    
    img_paths = []
    img_names = []
    for path, subdirs, files in chain.from_iterable(os.walk(dir) for dir in file_path):
        for name in files:
            if(name.endswith(".svs")):
                img_paths.append(os.path.join(path, name))
                img_names.append(name.split(".")[0])
    
    return img_paths, img_names


def extract_patches(img_paths, save_dir):
    """Extract valid patches from WSI images by applying filters and scoring informative patches

    :param img_path : list of file paths
    :param save_dir : path to save patches as .png file.
    """
    
    for index in range(626,len(img_paths)):
        data_path = img_paths[index]

        try:
            slide = open_slide(data_path)
        except:
            print("OpenSlide cannot open the file")
            continue
    
        file_name = data_path.split("/")[-1].split(".")[0] 
        print("Processing {} WSI at index {}".format(file_name,index))
        if(LOW_RES):
            level = slide.level_count-1
            downsample_factor = slide.level_downsamples[level]
        else:
            level = slide.get_best_level_for_downsample(SCALE_FACTOR)
            
        # Create thumbnail for the WSI at low resolution 
        thumbnail = slide.get_thumbnail((slide.dimensions[0] / PATCH_SIZE, slide.dimensions[1] / PATCH_SIZE))
        thum = np.array(thumbnail)
    
    
        # Read the whole slide images and convert in RGB, HSV and Grayscale format
        wsi_image, rgba_image, (slide_w_, slide_h_) = read_wsi(tif_file_path=data_path,level =level)
        wsi_rgb_, wsi_gray_, wsi_hsv_ = construct_colored_wsi(rgba_image)
    
    
        # Apply image filter on the rgb image
        filtered_img = filters.apply_image_filters(wsi_rgb_, info=None, save=False, display=False)
        # Otsu on grayscale
        otsu_mask = mask_rgb(filtered_img, filters.filter_otsu_threshold(filters.filter_rgb_to_grayscale(filtered_img), output_type="bool"))
    
    
        # Get contours, mask and bounding boxes for the filtered image 
        bounding_boxes, contour_coords, contours, mask = get_contours(img_as_ubyte(filters.rgb2gray(otsu_mask)),wsi_rgb_.shape)

        # Extract patches at given patch size
        patches, patches_coords = construct_bags(wsi_image,wsi_rgb_,
                                                 contours, mask, 
                                                level=level,
                                                PATCH_SIZE =PATCH_SIZE)
        # scoring individual patches
        if(len(patches)>0):
            all_tiles = []
            tile_scores = []
            for idx,patch in enumerate(patches):
                blur_map, score, isBlur = estimate_blur(patch,threshold=1000)
                if(not isBlur):
                    tissue_pcnt = tissue_percent(patch)
                    score, color_factor, s_and_v_factor, quantity_factor = score_tile(patch,tissue_pcnt)
                    all_tiles.append(patch)
                    tile_scores.append(score)

            tiles_df = pd.Series(all_tiles)
            sort_index = np.array(np.argsort(tile_scores)[::-1]) # reverse sorting                        
            tiles_by_score = tiles_df.reset_index(drop=True)[sort_index]
        else :
            continue
    
    
        # Save the tiles
        png_dir = os.path.join(save_dir,file_name)
        if not os.path.exists(os.path.join(BASE_PATH,png_dir)):
            os.mkdir(os.path.join(BASE_PATH,png_dir))

        Image.fromarray(thum).save(os.path.join(BASE_PATH+png_dir+"/" +"thumbnail" + ".png"))
        num_patches = min(len(patches),10)
        for idx in range(num_patches):
            img = tiles_by_score[idx]
            pil_img = Image.fromarray(img)
            fullpath = os.path.join(BASE_PATH+png_dir+"/" +str(idx) + ".png")
            pil_img.save(fullpath,'PNG')
        
    
def __init__():
    """
    Execute the script
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=Path, help="Path to input dir containing WSI images")
    parser.add_argument("save_path", type=Path, help="Path to save extracted patches")

    arguments = parser.parse_args()    
    if os.path.exists(arguments.file_path):
        print("File exist")
        img_path_list = get_files(file_path= arguments.file_path)
        extract_patches(img_paths=img_path_list, save_dir= arguments.save_path)

   
        
