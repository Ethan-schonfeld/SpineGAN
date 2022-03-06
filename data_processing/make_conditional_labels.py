#!/usr/bin/env python
# coding: utf-8

# In[1]:


image_directory = "/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/abnormality_conditional_dataset/"


# In[3]:


import os
import json
import sys

# In[ ]:


labels = {}
image_names = list(os.listdir(image_directory))
for picture_name in normal_names:
    print(picture_name[-4:])
    exit(0)
    labels[picture_name] = 0

# In[ ]:


with open("../stylegan2-ada-pytorch-main/abnormality_labels.json", "w") as outfile:
    json.dump(labels, outfile)

