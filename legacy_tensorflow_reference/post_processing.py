# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 21:41:49 2022

@author: Rashid Rao
"""
import numpy as np
from skimage.morphology import reconstruction
from skimage.morphology import remove_small_objects
from skimage import color, morphology
from skimage.measure import label, regionprops
import math
import skimage.measure as measure
from skimage import util

def dilate_this(image_src, dilation_level=1):
    # setting the dilation_level
    dilation_level = 3 if dilation_level < 3 else dilation_level
    
    # obtain the kernel by the shape of (dilation_level, dilation_level)
    structuring_kernel = np.full(shape=(dilation_level, dilation_level), fill_value=255)
    
    orig_shape = image_src.shape
    pad_width = dilation_level - 2
    
    # pad the image with pad_width
    image_pad = np.pad(array=image_src, pad_width=pad_width, mode='constant')
    pimg_shape = image_pad.shape
    h_reduce, w_reduce = (pimg_shape[0] - orig_shape[0]), (pimg_shape[1] - orig_shape[1])
    
    # obtain the submatrices according to the size of the kernel
    flat_submatrices = np.array([
        image_pad[i:(i + dilation_level), j:(j + dilation_level)]
        for i in range(pimg_shape[0] - h_reduce) for j in range(pimg_shape[1] - w_reduce)
    ])
    
    # replace the values either 255 or 0 by dilation condition
    image_dilate = np.array([255 if (i == structuring_kernel).any() else 0 for i in flat_submatrices])
    # obtain new matrix whose shape is equal to the original image size
    image_dilate = image_dilate.reshape(orig_shape)
    
    return image_dilate

def fun_post_processing(y_bin,y_pred_bin):
    #%% Post Processing
    # Copy the thresholded image.
    y_pred_bin_pp = y_pred_bin.copy()
    seed = np.copy(y_pred_bin_pp)
    seed[1:-1, 1:-1] = y_pred_bin_pp.max()
    mask = y_pred_bin_pp
    # Hole Filling
    y_pred_bin_pp = reconstruction(seed, mask, method='erosion')
    
    ## Find Biggest Region
    label_img = label(y_pred_bin_pp)
    regions = regionprops(label_img.astype('int'))
    
    #print('')
    #print('No of reigons in this image: ',len(regions))
    
    ## GETTING BLOBS USING REGION PROP
    obj_area = [prop.area for prop in regions]
    ## SORING BLOBS ACCORDING TO AREA
    obj_area_sorted = sorted(obj_area)
    ## Getting Index For Largest Blob
    tmp = max(obj_area)
    index = obj_area.index(tmp)
    
    # Getting Largest Blob
    ree= regions[index].filled_image.astype('int8')*255
   # Getting Location for Largest Blob
    xx  = regions[index].bbox[1]
    yy  = regions[index].bbox[0]
    
    post_processed=np.zeros((y_bin.shape[0],y_bin.shape[1]))
    
    post_processed[yy: yy + ree.shape[0], xx: xx + ree.shape[1]] = ree

    ## DILATION
    #post_processed = dilate_this(post_processed)
    return post_processed