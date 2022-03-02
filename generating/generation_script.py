#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle


# In[3]:


checkpoint_path = "~/cs236g/SpineGAN/stylegan2-ada-pytorch-main/training-ab-cond-ada/00002-abnormality_conditional_dataset-cond-auto1-kimg60-ada-target0.6-bgcfnc/network-snapshot-000060.pkl"


# In[5]:


save_folder = "~/cs236g/generated_normal"


# In[6]:


try:
    with open(local_checkpoint_path, 'rb') as f:
        G = pickle.load(f)['G_ema']  # torch.nn.Module
    print("Worked")
except:
    print("Failed")


# In[ ]:


z = torch.randn([1, G.z_dim])    # latent codes
c = 0                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]
print(img.size())

