#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
import numpy as np
import cv2
from PIL import Image


# In[33]:


def crop_square(image_directory=None, image=None):
    if image is None:
        image = np.load(image_directory)
    smallest_dimension_size = min(image.shape[0], image.shape[1])
    largest_dimension_size = max(image.shape[0], image.shape[1])
    amount_to_cut_from_each_side = int((largest_dimension_size - smallest_dimension_size)/2)
    if image.shape[0] <= image.shape[1]:
        square_image = image[:, amount_to_cut_from_each_side:(amount_to_cut_from_each_side+image.shape[0])]
    else:
        square_image = image[amount_to_cut_from_each_side:(amount_to_cut_from_each_side+image.shape[1]), :]
    return square_image


# In[48]:


def downsample_square_image(image_directory=None, image=None, desired_size=128):
    if image is None:
        image = np.load(image_directory)
    res = cv2.resize(image, dsize=(desired_size, desired_size), interpolation=cv2.INTER_NEAREST)
    return res


# In[57]:


def save_as_png(image_directory=None, image=None, file_directory="./train_images/"):
    if image is None:
        image = np.load(image_directory)
    im = Image.fromarray(image)
    im.save(file_directory)


# In[66]:


def read_png_to_npy(file_directory):
    with Image.open(file_directory) as im:
        img_np = np.asarray(im)
        return img_np

