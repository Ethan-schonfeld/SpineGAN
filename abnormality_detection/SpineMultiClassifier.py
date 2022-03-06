#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0), ratio=(1., 1.))
])


# In[ ]:


checkpoint_path = "/home/ethanschonfeld/cs236g/SpineGAN/abnormality_detection/"


# In[ ]:


train_label_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/train.csv"
train_image_directory = "/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/abnormality_conditional_dataset/"
test_label_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/test.csv"
test_image_directory = "/home/ethanschonfeld/cs236g/test_dataset/"


# In[ ]:


print("Loading in train images and labels")
train_annotations = pd.read_csv(train_label_directory)
train_image_file_list = list(os.listdir(train_image_directory))
os.chdir(train_image_directory)
train_images = []
train_labels = []
for filename in train_image_file_list:
    extension = filename[-4:]
    if extension != ".png":
        continue
    try:
        image = Image.open(filename)
        label = np.zeros(8)
        for idx_number in train_annotations.index:
            if train_annotations.loc[idx_number, "image_id"] == filename[:-4]:
                lesion_type = train_annotations.loc[idx_number, "lesion_type"]
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
        train_images.append(three_channel_data)
        train_labels.append(label)
    except:
        print(filename)
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
print("Loaded!")


# In[ ]:


print("Loading in test images and labels")
test_annotations = pd.read_csv(test_label_directory)
test_image_file_list = list(os.listdir(test_image_directory))
os.chdir(test_image_directory)
test_images = []
test_labels = []
for filename in test_image_file_list:
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
    except:
        print(filename)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
print("Loaded!")


# In[ ]:


# compute class weights for each category


# In[ ]:


class_weights_0 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,0]).tolist(), y=train_labels[:,0])
class_weights_0 = {i : class_weights_0[i] for i in range(2)}


# In[ ]:


class_weights_1 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,1]).tolist(), y=train_labels[:,1])
class_weights_1 = {i : class_weights_1[i] for i in range(2)}


# In[ ]:


class_weights_2 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,2]).tolist(), y=train_labels[:,2])
class_weights_2 = {i : class_weights_2[i] for i in range(2)}


# In[ ]:


class_weights_3 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,3]).tolist(), y=train_labels[:,3])
class_weights_3 = {i : class_weights_3[i] for i in range(2)}


# In[ ]:


class_weights_4 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,4]).tolist(), y=train_labels[:,4])
class_weights_4 = {i : class_weights_4[i] for i in range(2)}


# In[ ]:


class_weights_5 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,5]).tolist(), y=train_labels[:,5])
class_weights_5 = {i : class_weights_5[i] for i in range(2)}


# In[ ]:


class_weights_6 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,6]).tolist(), y=train_labels[:,6])
class_weights_6 = {i : class_weights_6[i] for i in range(2)}


# In[ ]:


class_weights_7 = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels[:,7]).tolist(), y=train_labels[:,7])
class_weights_7 = {i : class_weights_7[i] for i in range(2)}


# In[ ]:


model = models.densenet201(pretrained=True)
model = nn.Sequential(
    model,
    nn.Linear(in_features=1000, out_features=8, bias=True),
    nn.Sigmoid()
)


# In[ ]:


if torch.cuda.is_available():
    model.to('cuda')


# In[ ]:


def BCELoss_class_weighted(weights):

    def loss(inputs, target):
        inputs = torch.clamp(inputs,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(inputs) - (1 - target) * weights[0] * torch.log(1 - inputs)
        return torch.mean(bce)

    return loss


# In[ ]:


criterion_0 = BCELoss_class_weighted(class_weights_0)
criterion_1 = BCELoss_class_weighted(class_weights_1)
criterion_2 = BCELoss_class_weighted(class_weights_2)
criterion_3 = BCELoss_class_weighted(class_weights_3)
criterion_4 = BCELoss_class_weighted(class_weights_4)
criterion_5 = BCELoss_class_weighted(class_weights_5)
criterion_6 = BCELoss_class_weighted(class_weights_6)
criterion_7 = BCELoss_class_weighted(class_weights_7)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[1]:


# About 42 min per epoch on CPU
batch_size = 32
num_batches = math.ceil(train_images.shape[0]/batch_size)
best_test_auc_estimate = 0

for i in range(0, 10000): # they used 10000
    epoch_loss = 0.0
    epoch_auc_estimation_0 = []
    epoch_auc_estimation_1 = []
    epoch_auc_estimation_2 = []
    epoch_auc_estimation_3 = []
    epoch_auc_estimation_4 = []
    epoch_auc_estimation_5 = []
    epoch_auc_estimation_6 = []
    epoch_auc_estimation_7 = []
    
    for batch_num in range(num_batches):
        model.train()
        optimizer.zero_grad()
        
        if batch_num % 5 == 0:
            print("Epoch: ", i, " Batch: ", batch_num)
            
        batch = train_images[batch_num*batch_size:(batch_num+1)*batch_size, :, :, :]
        batch_labels = train_labels[batch_num*batch_size:(batch_num+1)*batch_size]
        batch_labels = torch.Tensor(batch_labels)
        
        train_X = torch.empty(0, 3, 224, 224)
        for image_number in range(batch.shape[0]):
            tensor = preprocess(Image.fromarray(batch[image_number, :, :, :]))
            tensor = tensor.unsqueeze(0)
            train_X = torch.cat([train_X, tensor], dim=0)
        if torch.cuda.is_available():
            train_X.to('cuda')
            
        # augment the images according to the augmentation defined above
        train_X = transform_augment(train_X).to('cuda')
            
        outputs = model(train_X).to('cuda')
        loss_0 = criterion(outputs[:,0].to('cuda'), batch_labels[:,0].unsqueeze(1).to('cuda'))
        loss_1 = criterion(outputs[:,1].to('cuda'), batch_labels[:,1].unsqueeze(1).to('cuda'))
        loss_2 = criterion(outputs[:,2].to('cuda'), batch_labels[:,2].unsqueeze(1).to('cuda'))
        loss_3 = criterion(outputs[:,3].to('cuda'), batch_labels[:,3].unsqueeze(1).to('cuda'))
        loss_4 = criterion(outputs[:,4].to('cuda'), batch_labels[:,4].unsqueeze(1).to('cuda'))
        loss_5 = criterion(outputs[:,5].to('cuda'), batch_labels[:,5].unsqueeze(1).to('cuda'))
        loss_6 = criterion(outputs[:,6].to('cuda'), batch_labels[:,6].unsqueeze(1).to('cuda'))
        loss_7 = criterion(outputs[:,7].to('cuda'), batch_labels[:,7].unsqueeze(1).to('cuda'))
        loss = loss_0+loss_1+loss_2+loss_3+loss_4+loss_5+loss_6+loss_7
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        train_auc_0 = roc_auc_score(batch_labels[:,0].cpu().detach().numpy(), outputs[:,0].cpu().detach().numpy())
        epoch_auc_estimation_0.append(train_auc_0)
        train_auc_1 = roc_auc_score(batch_labels[:,1].cpu().detach().numpy(), outputs[:,1].cpu().detach().numpy())
        epoch_auc_estimation_1.append(train_auc_1)
        train_auc_2 = roc_auc_score(batch_labels[:,2].cpu().detach().numpy(), outputs[:,2].cpu().detach().numpy())
        epoch_auc_estimation_2.append(train_auc_2)
        train_auc_3 = roc_auc_score(batch_labels[:,3].cpu().detach().numpy(), outputs[:,3].cpu().detach().numpy())
        epoch_auc_estimation_3.append(train_auc_3)
        train_auc_4 = roc_auc_score(batch_labels[:,4].cpu().detach().numpy(), outputs[:,4].cpu().detach().numpy())
        epoch_auc_estimation_4.append(train_auc_4)
        train_auc_5 = roc_auc_score(batch_labels[:,5].cpu().detach().numpy(), outputs[:,5].cpu().detach().numpy())
        epoch_auc_estimation_5.append(train_auc_5)
        train_auc_6 = roc_auc_score(batch_labels[:,6].cpu().detach().numpy(), outputs[:,6].cpu().detach().numpy())
        epoch_auc_estimation_6.append(train_auc_6)
        train_auc_7 = roc_auc_score(batch_labels[:,7].cpu().detach().numpy(), outputs[:,7].cpu().detach().numpy())
        epoch_auc_estimation_7.append(train_auc_7)
        
    print("Epoch ", i, " Train AUC estimation No finding: ", sum(epoch_auc_estimation_0)/len(epoch_auc_estimation_0))
    print("Epoch ", i, " Train AUC estimation Disc space narrowing: ", sum(epoch_auc_estimation_1)/len(epoch_auc_estimation_1))
    print("Epoch ", i, " Train AUC estimation Foraminal stenosis: ", sum(epoch_auc_estimation_2)/len(epoch_auc_estimation_2))
    print("Epoch ", i, " Train AUC estimation Osteophytes: ", sum(epoch_auc_estimation_3)/len(epoch_auc_estimation_3))
    print("Epoch ", i, " Train AUC estimation Spondylolysthesis: ", sum(epoch_auc_estimation_4)/len(epoch_auc_estimation_4))
    print("Epoch ", i, " Train AUC estimation Surgical implant: ", sum(epoch_auc_estimation_5)/len(epoch_auc_estimation_5))
    print("Epoch ", i, " Train AUC estimation Vertebral collapse: ", sum(epoch_auc_estimation_6)/len(epoch_auc_estimation_6))
    print("Epoch ", i, " Train AUC estimation Other lesions: ", sum(epoch_auc_estimation_7)/len(epoch_auc_estimation_7))
    print("Epoch ", i, " Train AUC estimation Average: ", (sum(epoch_auc_estimation_0)+sum(epoch_auc_estimation_1)+sum(epoch_auc_estimation_2)+sum(epoch_auc_estimation_3)+sum(epoch_auc_estimation_4)+sum(epoch_auc_estimation_5)+sum(epoch_auc_estimation_6)+sum(epoch_auc_estimation_7))/len(epoch_auc_estimation_0)*8)

                
    print("Epoch ", i, " loss: ", epoch_loss)
    # get random sample of 600 of test images
    total_num_test = int(test_images.shape[0])
    # sample 600 without replacement
    total_num_test = range(total_num_test)
    random_sample = random.sample(total_num_test, 600)
    test_batch = test_images[random_sample, :, :, :]
    test_batch_labels = [test_labels[i] for i in random_sample]
    test_batch_labels = torch.Tensor(test_batch_labels)
    
    test_X = torch.empty(0, 3, 224, 224)
    for image_number in range(test_batch.shape[0]):
        tensor = preprocess(Image.fromarray(test_batch[image_number, :, :, :]))
        tensor = tensor.unsqueeze(0)
        test_X = torch.cat([test_X, tensor], dim=0)
    if torch.cuda.is_available():
        test_X.to('cuda')
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X.to('cuda'))
        print("AUC No Finding: ")
        test_auc_0 = roc_auc_score(test_batch_labels[:,0].cpu().detach().numpy(), test_outputs[:,0].cpu().detach().numpy())
        print(test_auc_0)
        
        print("AUC Disc space narrowing: ")
        test_auc_1 = roc_auc_score(test_batch_labels[:,1].cpu().detach().numpy(), test_outputs[:,1].cpu().detach().numpy())
        print(test_auc_1)
        
        print("Foraminal stenosis: ")
        test_auc_2 = roc_auc_score(test_batch_labels[:,2].cpu().detach().numpy(), test_outputs[:,2].cpu().detach().numpy())
        print(test_auc_2)
        
        print("AUC Osteophytes: ")
        test_auc_3 = roc_auc_score(test_batch_labels[:,3].cpu().detach().numpy(), test_outputs[:,3].cpu().detach().numpy())
        print(test_auc_3)
        
        print("AUC Spondylolysthesis: ")
        test_auc_4 = roc_auc_score(test_batch_labels[:,4].cpu().detach().numpy(), test_outputs[:,4].cpu().detach().numpy())
        print(test_auc_4)
        
        print("AUC Surgical implant: ")
        test_auc_5 = roc_auc_score(test_batch_labels[:,5].cpu().detach().numpy(), test_outputs[:,5].cpu().detach().numpy())
        print(test_auc_5)
        
        print("AUC Vertebral collapse: ")
        test_auc_6 = roc_auc_score(test_batch_labels[:,6].cpu().detach().numpy(), test_outputs[:,6].cpu().detach().numpy())
        print(test_auc_6)
        
        print("AUC Other lesions: ")
        test_auc_7 = roc_auc_score(test_batch_labels[:,7].cpu().detach().numpy(), test_outputs[:,7].cpu().detach().numpy())
        print(test_auc_7)
        
    print("Epoch ", i, " Test Sample AUC: ", (test_auc_0+test_auc_1+test_auc_2+test_auc_3+test_auc_4+test_auc_5+test_auc_6+test_auc_7)/8)
    torch.save(model, checkpoint_path+"checkpoint_cond_aug_"+str(i)+".pt")
    if test_auc > best_test_auc_estimate:
        best_test_auc_estimate = test_auc
    #    torch.save(model, checkpoint_path+"checkpoint_aug_"+str(i)+".pt")


# In[ ]:




