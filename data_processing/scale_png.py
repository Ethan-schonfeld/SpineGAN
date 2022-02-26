#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import math


# In[ ]:


directory = "/home/ethanschonfeld/cs236g/stylegan2-ada-pytorch-main/normal_dataset/00000"


# In[ ]:


# the purpose of this is to change the scale of the image to be of maximum pixel intensity of 256
os.chdir(directory)
dir_list = list(os.listdir(directory))
for image_dir in dir_list:
    if image_dir[-3:] == "png":
        # load image in
        image = Image.open(image_dir)
        data = np.asarray(image)
        # get maximum pixel intensity
        max_intensity = np.amax(data)
        # how many times does 256 fit in
        scale_factor = math.ceil(max_intensity / 256)
        # now apply the scale factor
        scaled_data = data // scale_factor
        im = Image.fromarray(scaled_data)
        im.save(image_dir)

