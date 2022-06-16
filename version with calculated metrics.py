#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import numpy as np
from numpy import ravel
import itertools
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import math
from config import *
from Utils_1 import *


# In[2]:


data=pd.read_excel(DATA_PATH/'data.xlsx'")


# In[3]:


data = download_prepare_data(data)


# In[4]:


#separate in 10 second intervals
data_dictionary=expand_data(data, f'{SENSORS[0]}_raw', 10)
for j in SENSORS[1:]:
    data_dictionary[f'{j}_raw']=expand_data(data, f'{j}_raw', 10)
    


# In[5]:


data_dictionary['mag_raw']=[[np.sqrt((x**2+y**2+z**2)) for x,y,z in zip(data_dictionary.x_raw.iloc[i],
        data_dictionary.y_raw.iloc[i], data_dictionary.z_raw.iloc[i])] for i in range(0, len(data_dictionary))]


# In[6]:


data_dictionary['mag1_raw']=[[np.sqrt((x**2+z**2)) for x,z in zip(data_dictionary.x_raw.iloc[i],
        data_dictionary.z_raw.iloc[i])] for i in range(0, len(data_dictionary))]


# In[7]:



sensors= SENSORS1


# In[8]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(median(data_dictionary,sensors)))


# In[9]:


data_f.head()


# In[10]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_median(data_dictionary,sensors)))


# In[11]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_mean(data_dictionary,sensors)))


# In[12]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(var(data_dictionary,sensors)))


# In[13]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_var(data_dictionary,sensors)))


# In[14]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(std(data_dictionary,sensors)))


# In[15]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_std(data_dictionary,sensors)))


# In[16]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(variation(data_dictionary,sensors)))


# In[17]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_variation(data_dictionary,sensors)))


# In[18]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(skw(data_dictionary,sensors)))


# In[19]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_skw(data_dictionary,sensors)))


# In[20]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(skew(data_dictionary,sensors)))


# In[21]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_skew(data_dictionary,sensors)))


# In[22]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(krt(data_dictionary,sensors)))


# In[23]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_krt(data_dictionary,sensors)))


# In[24]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(kurtosis(data_dictionary,sensors)))


# In[25]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_kurtosis(data_dictionary,sensors)))


# In[26]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(tvar(data_dictionary,sensors)))


# In[27]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_tvar(data_dictionary,sensors)))


# In[28]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_kstat(data_dictionary,sensors)))


# In[29]:


data_f=pd.DataFrame()
for i in range(0, len(data_dictionary)):
    data_f=data_f.append(pd.DataFrame.from_dict(calculate_kstatvar(data_dictionary,sensors)))



                   




