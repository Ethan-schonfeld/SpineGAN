#!/usr/bin/env python
# coding: utf-8

# In[1]:


image_directory = "/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/conditional_dataset/"
label_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/train.csv"

# In[3]:


import os
import json
import sys
import pandas as pd

# In[ ]:

annotations = pd.read_csv(label_directory)
annotations.index = annotations.loc[:, "image_id"]

labels = {}
image_names = list(os.listdir(image_directory))
for picture_name in image_names:
    print(picture_name[-4:])
    image_id = picture_name[:-4]
    print(image_id)
    exit(0)
    if picture_name[-4:] == ".png":
        pass
    exit(0)
    labels[picture_name] = 0

# In[ ]:


with open("../stylegan2-ada-pytorch-main/abnormality_labels.json", "w") as outfile:
    json.dump(labels, outfile)

