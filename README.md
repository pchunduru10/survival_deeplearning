# Prognostic Risk Stratification of Gliomas Using Deep Learning in Digital Pathology Images

## Overview : 
The project-codebase demonstrates the training and analysis of H&E stained histopathology images from low and high grade brain tumor patient cohort. Digitized whole slide image were obtained from external multi institutional database. The project structure showcases
* Processing of  high resolution whole slide images (WSI) (20x magnification)
    - Downsampling and Sclaing of WSI form 20x by a factor.
    - Applying combination of image based filters.
    - Determine Countours, Mask and Extract Patches.
    - Scoring of Patches/tiles based on multiple qualitative and quantitative factors.
* Develop ResNet based CNN architecture to train on the extracted patches
    - Image resolution at 1024x1024
    - Custom Loss Function
    - Fine tune layers
    - Regression output in defined range.
    - tensoboard visualization for performance
    
NOTE: Data is not provided in the repo. Train a neural network on your dataset by setting up in the format as give in `main.py` 

    
