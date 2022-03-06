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
import sys
import random
import json
import matplotlib.pyplot as plt


# In[ ]:


checkpoint_path = "/home/ethanschonfeld/cs236g/SpineGAN/abnormality_classification/"


# In[ ]:


train_abnormality_directory = "/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/abnormality_conditional_dataset/dataset.json"
train_image_directory = "/home/ethanschonfeld/cs236g/SpineGAN/stylegan2-ada-pytorch-main/abnormality_conditional_dataset/"

train_gen_normal_directory = "/home/ethanschonfeld/cs236g/SpineGAN/generated/normal/"
train_gen_abnormal_directory = "/home/ethanschonfeld/cs236g/SpineGAN/generated/abnormal/"


# In[ ]:


test_abnormality_directory = "/home/ethanschonfeld/cs236g/vindr/annotations/test.csv"
test_image_directory = "/home/ethanschonfeld/cs236g/test_dataset/"


# In[ ]:


# these are the required transforms for Pytorch densenet model.
# we will apply these to all training and testing images
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# comment this if you don't want augmentation

transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=(-5, 5)),
    transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0), ratio=(1., 1.))
])

# In[ ]:


# load in train abnormality labels
f = open(train_abnormality_directory)
train_abnormality_labels = json.load(f)
f.close()


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


# load in train image data and make labels
train_image_file_list = list(os.listdir(train_image_directory))
os.chdir(train_image_directory)
train_labels = []
train_images = []
for filename in train_image_file_list:
    extension = filename[-4:]
    if extension != ".png":
        continue
    try:
        label = train_abnormality_labels[filename]
        train_labels.append(label)
        image = Image.open(filename)
        data = np.asarray(image)
        three_channel_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
        train_images.append(three_channel_data)
    except:
        print(filename)

# load in train gen image data and make labels
train_gen_normal_file_list = list(os.listdir(train_gen_normal_directory))
os.chdir(train_gen_normal_directory)
for filename in train_gen_normal_file_list:
    extension = filename[-4:]
    if extension != ".png":
        continue
    train_labels.append(0)
    image = Image.open(filename)
    data = np.asarray(image)
    three_channel_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    train_images.append(three_channel_data)
    
print("Made normal images")
    
train_gen_abnormal_file_list = list(os.listdir(train_gen_abnormal_directory))
os.chdir(train_gen_abnormal_directory)
for filename in train_gen_abnormal_file_list:
    extension = filename[-4:]
    if extension != ".png":
        continue
    train_labels.append(1)
    image = Image.open(filename)
    data = np.asarray(image)
    three_channel_data = np.repeat(data[:, :, np.newaxis], 3, axis=2)
    train_images.append(three_channel_data)
    
print("Made abnormal images")
    
train_images = np.asarray(train_images)
print("Finalized training images")



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


# compute class weights
class_weights = class_weight.compute_class_weight("balanced", 
                                                   classes=np.unique(train_labels).tolist(), y=train_labels)
class_weights = {i : class_weights[i] for i in range(2)}


# In[ ]:


model = models.densenet121(pretrained=True)
model = nn.Sequential(
    model,
    nn.Linear(in_features=1000, out_features=1, bias=True),
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


criterion = BCELoss_class_weighted(class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


# About 42 min per epoch on CPU
batch_size = 32
num_batches = math.ceil(train_images.shape[0]/batch_size)
best_test_auc_estimate = 0

for i in range(0, 10000): # they used 10000
    epoch_loss = 0.0
    epoch_auc_estimation = []
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
        #train_X = transform_augment(train_X).to('cuda')
            
        outputs = model(train_X.to('cuda'))
        loss = criterion(outputs.to('cuda'), batch_labels.unsqueeze(1).to('cuda'))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        train_auc = roc_auc_score(batch_labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        epoch_auc_estimation.append(train_auc)
        
    print("Epoch ", i, " Train AUC estimation: ", sum(epoch_auc_estimation)/len(epoch_auc_estimation))
                
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
        test_auc = roc_auc_score(test_batch_labels.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())
    print("Epoch ", i, " Test Sample AUC: ", test_auc)
    torch.save(model, checkpoint_path+"checkpoint_gan_"+str(i)+".pt") # change name to include aug if using augmentation
    if test_auc > best_test_auc_estimate:
        best_test_auc_estimate = test_auc
    #    torch.save(model, checkpoint_path+"checkpoint_aug_"+str(i)+".pt")

