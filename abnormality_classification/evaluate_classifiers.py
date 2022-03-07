#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
import os
import random
import json
import matplotlib.pyplot as plt


# In[ ]:


checkpoint_path = "/home/ethanschonfeld/cs236g/SpineGAN/abnormality_classification/"


# In[ ]:


test_abnormality_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/test.csv"
test_image_directory = "/home/ethanschonfeld/cs236g/test_dataset/"


# In[ ]:


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# In[ ]:


# load in test abnormality labels
test_abnormality_dataset = pd.read_csv(test_abnormality_directory)
test_abnormality_labels = {}
for index_number in (test_abnormality_dataset.index):
    image_id = test_abnormality_dataset.loc[index_number, "image_id"]
    image_file_name = image_id+".png"
    label = test_abnormality_dataset.loc[index_number, "lesion_type"]
    if label == "No finding":
        label = 0
    else:
        label = 1
    test_abnormality_labels[image_file_name] = label


# In[ ]:


# load in test image data and make labels
test_image_file_list = list(os.listdir(test_image_directory))
os.chdir(test_image_directory)
test_labels = []
test_images = []
for filename in test_image_file_list:
    extension = filename[-4:]
    if extension != ".png":
        continue
    try:
        label = test_abnormality_labels[filename]
        test_labels.append(label)
        image = Image.open(filename)
        data = np.array(image)
        three_channel_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        test_images.append(three_channel_data)
    except:
        print(filename)
test_images = np.asarray(test_images)


# In[ ]:


test_labels = torch.Tensor(test_labels)


# In[ ]:


test_X = torch.empty(0, 3, 224, 224)
for image_number in range(test_images.shape[0]):
    tensor = preprocess(Image.fromarray(test_images[image_number, :, :, :]))
    tensor = tensor.unsqueeze(0)
    test_X = torch.cat([test_X, tensor], dim=0)


# In[ ]:


os_list = list(os.listdir(checkpoint_path))
os.chdir(checkpoint_path)
for file_name in os_list:
    if file_name[:10] != "checkpoint":
        continue
    model = torch.load(file_name)
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X.to('cuda'))
        test_auc = roc_auc_score(test_labels.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
    print(file_name, " Test AUC: ", test_auc)

