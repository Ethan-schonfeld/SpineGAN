#!/usr/bin/env python
# coding: utf-8

# In[1]:


normal_data_source = "../stylegan2-ada-pytorch-main/normal_dataset"
abnormal_data_source = "../stylegan2-ada-pytorch-main/abnormal_dataset"


# In[3]:


import json


# In[ ]:


labels = {}
normal_names = list(os.listdir(normal_data_source))
for picture_name in normal_names:
    labels[picture_name] = 0
    
abnormal_names = list(os.listdir(abnormal_data_source))
for picture_name in abnormal_names:
    labels[picture_name] = 1


# In[ ]:


with open("../stylegan2-ada-pytorch-main/abnormality_labels.json", "w") as outfile:
    json.dump(labels, outfile)

