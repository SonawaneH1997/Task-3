#!/usr/bin/env python
# coding: utf-8

# # Task-3(To Explore Unsupervised Machine Learning)
# K-Means Clustering:To predict the optimum number of clusters and represent it visually.
# Presented By-Harshada Suresh Sonawane

# # K-Means Clustering

# In[3]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[4]:


#load the dataset
iris=datasets.load_iris()
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df.head()


# In[5]:


#Describe the data
iris_df.describe()


# In[6]:


fig=plt.figure(figsize=(10,10))
sns.heatmap(iris_df.corr(),linewidths=1, annot=True)


# In[7]:


#visualising relation between features
plt.scatter(x= 'sepal length (cm)', y='sepal width (cm)', data=iris_df, color='b')
plt.xlabel('sepal length (cm)', fontsize=15)
plt.ylabel('sepal width (cm)', fontsize=15)
plt.show


# In[6]:


#visualising relation between features
plt.scatter(x= 'petal length (cm)', y='petal width (cm)', data=iris_df, color='g')
plt.xlabel('petal length (cm)', fontsize=15)
plt.ylabel('petal width (cm)', fontsize=15)
plt.show


# In[7]:


#Finding k using elbow method
wcss=[]
k_range=10
for i in range (1, k_range):
    k=KMeans(i)
    k.fit(iris_df)
    w=k.inertia_
    wcss.append(w)
wcss


# In[8]:


clusters=range(1,k_range)
plt.plot(clusters, wcss, marker='.', color="b", markersize=10 )
plt.xlabel('No of clusters', fontsize=15 )
plt.ylabel('wcss', fontsize=15)
plt.title('The Elbow Method', fontsize=25)
plt.show()


# We can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as 3.

# In[9]:


kmeans= KMeans(3)
kmeans.fit(iris_df)


# In[10]:


centers=kmeans.cluster_centers_
print(centers)


# In[12]:


iris_pred=iris_df.copy()
iris_pred['predicted']=kmeans.fit_predict(iris_df)


# In[13]:


#visualizing cluster according to predicted value for sepal length and sepal width
plt.scatter(x='sepal length (cm)', y='sepal width (cm)', c='predicted', cmap='rainbow', data=iris_pred)
plt.xlabel('sepal length', fontsize=15)
plt.ylabel('sepal width', fontsize=15)
plt.title('k=3 sepal length vs sepal width', fontsize=25)
plt.show()


# In[14]:


#visualizing cluster according to predicted value for petal length and petal width
plt.scatter(x='petal length (cm)', y='petal width (cm)', c='predicted', cmap='jet', data=iris_pred)
plt.xlabel('petal length', fontsize=15)
plt.ylabel('petal width', fontsize=15)
plt.title('k=3 petal length vs petal width', fontsize=25)
plt.show()


# In[16]:


#visualizing cluster for true labels
plt.scatter(x='sepal length (cm)', y='sepal width (cm)', c=iris.target, cmap='rainbow', data=iris_df)
plt.xlabel('sepal length', fontsize=15)
plt.ylabel('sepal width', fontsize=15)
plt.title('sepal length vs sepal width', fontsize=25)
plt.show()


# In[8]:


#visualizing cluster for true labels
plt.scatter(x='petal length (cm)', y='petal width (cm)', c=iris.target, cmap='jet', data=iris_df)
plt.xlabel('petaal length', fontsize=15)
plt.ylabel('petal width', fontsize=15)
plt.title('petal length vs petal width', fontsize=25)
plt.show()


# Visualizing the cluster for true values we can say that k=3 is appropriate solution for iris dataset.
