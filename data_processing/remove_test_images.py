#!/usr/bin/env python
# coding: utf-8

# In[1]:


test_annotations_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/test.csv"
training_images_dir = "/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/abnormality_conditional_dataset"


# In[4]:


import pandas as pd
import os


# In[7]:


# get names of all image_id for test set
test = pd.read_csv(test_annotations_directory)
all_test_ids = test.loc[:, "image_id"]
all_test_ids = list(set(all_test_ids))


# In[ ]:


# remove from GAN training folder all studies that are in the test set
os.chdir(training_images_dir)
image_list = list(os.listdir(training_images_dir))
count = 0
for image_file in image_list:
    count += 1
    image_id = image_file[:-4]
    if image_id in all_test_ids:
        os.remove(image_file)
        print("Removed: ", image_id)
    if count % 500 == 0:
        print("Number of iterations passed: ", count)

