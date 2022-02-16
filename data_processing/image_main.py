#!/usr/bin/env python
# coding: utf-8

# In[3]:


local_directory = "/Users/ethanschonfeld/desktop/CS236G/data/test_images/"


# In[4]:


virtual_directory = "/home/ethanschonfeld/cs236g/vindr/test_images/"


# In[ ]:


vindr_directory = "/home/ethanschonfeld/cs236g/vindr/"


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


annotations = pd.read_csv(vindr_directory+"annotations/test.csv")
annotations.index = annotations.loc[:, "image_id"]


# In[ ]:


for file in os.listdir(directory):
    if file == ".DS_Store":
        continue
    npy_image = convert_dicom_to_npy(directory+file, save=False)
    if npy_image is None:
        continue
    npy_image = crop_square(image=npy_image)
    npy_image = downsample_square_image(image=npy_image, desired_size=128)
    image_id = file[:,-6]
    if annotations[image_id, "lesion_type"] == "No finding":
        print("To normal:")
        save_as_png(image=npy_image, file_directory="/home/ethanschonfeld/cs236g/training_data/normal/"+image_id+".png")
    else:
        save_as_png(image=npy_image, file_directory="/home/ethanschonfeld/cs236g/training_data/abnormal/"+image_id+".png")
        print("To abnormal:")
        print("Saved: ", image_id)

