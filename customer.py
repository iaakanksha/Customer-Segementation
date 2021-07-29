#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


# In[12]:


df = pd.read_csv("Mallcustomers.csv")
df.head()


# In[13]:


df.columns


# In[14]:


df.shape


# In[15]:


df.info()


# In[16]:


df.isnull().sum()


# In[17]:


x=df.iloc[:,[3,4]].values


# In[18]:


print(x)


# In[ ]:





# In[20]:


wcss = []


for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_)


# sns.set()
# plt.plot(range(1,11), wcss)
# plt.title('The elbow point Graph')
# plt.xlabel("number of clusters")
# plt.ylabel("wcss")
# plt.show()

# In[28]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=1)
Y = kmeans.fit_predict(x)
print(Y)


# In[29]:


plt.figure(figsize=(8,8))
plt.scatter(x[Y==0,0], x[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(x[Y==1,0], x[Y==1,1], s=50, c='cyan', label='Cluster 2')
plt.scatter(x[Y==2,0], x[Y==2,1], s=50, c='yellow', label='Cluster 3')
plt.scatter(x[Y==3,0], x[Y==3,1], s=50, c='blue', label='Cluster 4')
plt.scatter(x[Y==4,0], x[Y==4,1], s=50, c='red', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='centroid')
plt.title('customer group')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.show()


# In[ ]:




