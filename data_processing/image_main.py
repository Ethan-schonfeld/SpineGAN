#!/usr/bin/env python
# coding: utf-8

# In[3]:


local_directory = "/Users/ethanschonfeld/desktop/CS236G/data/test_images/"


# In[4]:


virtual_directory = "/home/ethanschonfeld/cs236g/vindr/test_images/"


# In[6]:


from image_processing import *
import os
import torch


# In[ ]:


if torch.cuda.is_available():
    directory = virtual_directory
else:
    directory = local_directory


# In[ ]:


for file in os.listdir(directory):
    if file == ".DS_Store":
        continue
    npy_image = convert_dicom_to_npy(directory+file, save=True, save_directory="/home/ethanschonfeld/cs236g/training_data")

