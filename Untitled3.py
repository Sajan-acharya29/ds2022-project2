#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('911.csv.zip')


# In[2]:


df.head()


# In[15]:


extract=lambda x: x.split(':')[0]
df=df.assign(Reason=df['title'].apply(extract))


# In[16]:


sns.scatterplot(x = 'lat',y = 'lng',
               data = df, hue ='Reason')

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title('Location of 911 calls', fontdict = font1)
plt.ylabel('Longitude', fontdict = font2)
plt.xlabel('Latitude', fontdict = font2)
#This graph helps reperesent the relationship between location and reason for 911 call.


# In[ ]:




