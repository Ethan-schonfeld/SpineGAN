import math
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
import os
import sys
import random
import json
import matplotlib.pyplot as plt


# In[ ]:


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# In[ ]:


checkpoint_path = "/home/ethanschonfeld/cs236g/SpineGAN/abnormality_detection/"


test_label_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/test.csv"
test_image_directory = "/home/ethanschonfeld/cs236g/test_dataset/"


# In[ ]:


print("Loading in test images and labels")
test_annotations = pd.read_csv(test_label_directory)
test_image_file_list = list(os.listdir(test_image_directory))
os.chdir(test_image_directory)
test_images = []
test_labels = []
count = 0
for filename in test_image_file_list:
    print(count)
    extension = filename[-4:]
    if extension != ".png":
        continue
    try:
        image = Image.open(filename)
        label = np.zeros(8)
        for idx_number in test_annotations.index:
            if test_annotations.loc[idx_number, "image_id"] == filename[:-4]:
                lesion_type = test_annotations.loc[idx_number, "lesion_type"]
                if lesion_type == "No finding":
                    label[0] = 1
                elif lesion_type == "Disc space narrowing":
                    label[1] = 1
                elif lesion_type == "Foraminal stenosis":
                    label[2] = 1
                elif lesion_type == "Osteophytes":
                    label[3] = 1
                elif lesion_type == "Spondylolysthesis":
                    label[4] = 1
                elif lesion_type == "Surgical implant":
                    label[5] = 1
                elif lesion_type == "Vertebral collapse":
                    label[6] = 1
                elif lesion_type == "Other lesions":
                    label[7] = 1
        data = np.asarray(image)
        three_channel_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        test_images.append(three_channel_data)
        test_labels.append(label)
        count += 1
    except:
        print(filename)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
print("Loaded!")

test_X = torch.empty(0, 3, 224, 224)
for image_number in range(test_images.shape[0]):
    tensor = preprocess(Image.fromarray(test_images[image_number, :, :, :]))
    tensor = tensor.unsqueeze(0)
    test_X = torch.cat([test_X, tensor], dim=0)
if torch.cuda.is_available():
    test_X.to('cuda')
test_labels = torch.Tensor(test_labels)



# compute class weights for each category


# In[ ]:

files_list = os.listdir(checkpoint_path)
os.chdir(checkpoint_path)
for file in files_list:
    if file[0] != "c":
        continue
    model = torch.load(file)
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X.to('cuda'))
        try:
            test_auc_0 = roc_auc_score(test_batch_labels[:,0].cpu().detach().numpy(), test_outputs[:,0].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_1 = roc_auc_score(test_batch_labels[:,1].cpu().detach().numpy(), test_outputs[:,1].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_2 = roc_auc_score(test_batch_labels[:,2].cpu().detach().numpy(), test_outputs[:,2].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_3 = roc_auc_score(test_batch_labels[:,3].cpu().detach().numpy(), test_outputs[:,3].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_4 = roc_auc_score(test_batch_labels[:,4].cpu().detach().numpy(), test_outputs[:,4].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_5 = roc_auc_score(test_batch_labels[:,5].cpu().detach().numpy(), test_outputs[:,5].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_6 = roc_auc_score(test_batch_labels[:,6].cpu().detach().numpy(), test_outputs[:,6].cpu().detach().numpy())
        except:
            print("NaN")
        try:
            test_auc_7 = roc_auc_score(test_batch_labels[:,7].cpu().detach().numpy(), test_outputs[:,7].cpu().detach().numpy())
        except:
            print("NaN")
    try:    
        print(file, (test_auc_0+test_auc_1+test_auc_2+test_auc_3+test_auc_4+test_auc_5+test_auc_6+test_auc_7)/8)
        print(file, test_auc_0, test_auc_1, test_auc_2, test_auc_3, test_auc_4, test_auc_5, test_auc_6, test_auc_7)
        print("*"*50)
    except:
        print(file, "Test Sample AUC: NaN")
