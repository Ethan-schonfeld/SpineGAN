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
os.chdir(image_directory)
duplicates = []
for picture_name in image_names:
    image_id = picture_name[:-4]
    if picture_name[-4:] == ".png":
        condition = annotations.loc[image_id, "lesion_type"]
        if type(condition) == str:
            if condition == "No finding":
                labels[picture_name] = 0
            elif condition == "Disc space narrowing":
                labels[picture_name] = 1
            elif condition == "Foraminal stenosis":
                labels[picture_name] = 2
            elif condition == "Osteophytes":
                labels[picture_name] = 3
            elif condition == "Spondylolysthesis":
                labels[picture_name] = 4
            elif condition == "Surgical implant":
                labels[picture_name] = 5
            elif condition == "Vertebral collapse":
                labels[picture_name] = 6
            elif condition == "Other lesions":
                labels[picture_name] = 7
            else:
                os.remove(picture_name)
        elif len(set(condition)) == 1:
            condition = list(set(condition))[0]
            if condition == "No finding":
                labels[picture_name] = 0
            elif condition == "Disc space narrowing":
                labels[picture_name] = 1
            elif condition == "Foraminal stenosis":
                labels[picture_name] = 2
            elif condition == "Osteophytes":
                labels[picture_name] = 3
            elif condition == "Spondylolysthesis":
                labels[picture_name] = 4
            elif condition == "Surgical implant":
                labels[picture_name] = 5
            elif condition == "Vertebral collapse":
                labels[picture_name] = 6
            elif condition == "Other lesions":
                labels[picture_name] = 7
            else:
                os.remove(picture_name)
        else:
            duplicates.append(picture_name)

duplicates = list(set(duplicates))

for duplicate in duplicates:
    os.remove(duplicate)

# In[ ]:


with open("/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/conditional_dataset/dataset.json", "w") as outfile:
    json.dump(labels, outfile)

