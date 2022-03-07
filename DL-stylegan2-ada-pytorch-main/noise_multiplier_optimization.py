#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np

tf.get_logger().setLevel('ERROR')


# In[6]:


import tensorflow_privacy

from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


# In[7]:


x_noise = [(0.01)/1.5, 0.03/1.5, 0.06/1.5, 0.1/1.5, 0.2/1.5, 0.3/1.5, 0.4/1.5, 0.5/1.5, 0.6/1.5, 0.7/1.5, 0.8/1.5, 0.9/1.5, 1/1.5, 1.1/1.5, 1.2/1.5, 1.3/1.5, 1.4/1.5, 1.5/1.5, 1.6/1.5, 1.7/1.5, 1.8/1.5, 1.9/1.5, 2/1.5]
y_epsilon = []
for noise in x_noise:
    epsilon = compute_dp_sgd_privacy.compute_dp_sgd_privacy(n=3773,
                                              batch_size=1,
                                              noise_multiplier=noise,
                                              epochs=(2400000/3773),
                                              delta=(1/3773))
    epsilon = epsilon[0]
    y_epsilon.append(epsilon)


# In[10]:


plt.scatter(x_noise, np.log(y_epsilon))
plt.xlabel('Estimated Noise Multiplier (noise standard deviation/clipping norm)')
plt.ylabel('Log Base 10 Epsilon')


# In[ ]:




