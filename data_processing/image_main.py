#!/usr/bin/env python
# coding: utf-8

# In[3]:
import argparse

my_parser = argparse.ArgumentParser(description='Data Specifications')

# Add the arguments
my_parser.add_argument('DICOM',
                       help='the path to DICOM image folder')

my_parser.add_argument('labels',
                       help='the path to csv of DICOM image labels for condition')


# Execute the parse_args() method
args = my_parser.parse_args()

#local_directory = "/Users/ethanschonfeld/desktop/CS236G/data/train_images/"
directory = args.DICOM


# In[4]:


#virtual_directory = "/home/ethanschonfeld/cs236g/vindr/train_images/"


# In[ ]:


#vindr_directory = "/home/ethanschonfeld/cs236g/vindr/"
vindr_directory = args.labels

# In[6]:


from image_processing import *
import os
import pandas as pd
import torch


# In[ ]:


if torch.cuda.is_available():
    #directory = virtual_directory
    directory = directory
else:
    directory = local_directory


# In[ ]:


#annotations = pd.read_csv(vindr_directory+"annotations/train.csv")
annotations = pd.read_csv(vindr_directory)
annotations.index = annotations.loc[:, "image_id"]


# In[ ]:


for file in os.listdir(directory):
    print(file)
    if file == ".DS_Store":
        continue
    npy_image = convert_dicom_to_npy(directory+file, save=False)
    if npy_image is None:
        continue
    npy_image = crop_square(image=npy_image)
    npy_image = downsample_square_image(image=npy_image, desired_size=256)
    image_id = file[:-6]
    if bool(set(annotations.loc[:, "lesion_type"]) & set(annotations.loc[image_id, "lesion_type"])):
        print("To abnormal: ", image_id)
        save_as_png(image=npy_image, file_directory="/home/ethanschonfeld/cs236g/training_data/abnormal/"+image_id+".png")
    else:
        save_as_png(image=npy_image, file_directory="/home/ethanschonfeld/cs236g/training_data/normal/"+image_id+".png")
        print("To normal: ", image_id)

