#!/usr/bin/env python
# coding: utf-8

# In[36]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[3]:


train_1 = unpickle('data_batch_1')
train_2 = unpickle('data_batch_2')
train_3 = unpickle('data_batch_3')
train_4 = unpickle('data_batch_4')
train_5 = unpickle('data_batch_5')
test_data = unpickle('test_batch')


# In[18]:


labels = np.vstack((train_1[b'labels'], train_2[b'labels'], train_3[b'labels'], train_4[b'labels'], train_5[b'labels']))
data = np.vstack((train_1[b'data'], train_2[b'data'], train_3[b'data'], train_4[b'data'], train_5[b'data']))


# In[19]:


data = joblib.load('data.pkl')


# In[20]:


param_grid = {'C': [10] ,'gamma': [0.1,0.01,0.001],'kernel': ['rbf']}


# In[21]:


labels = labels.reshape(-1,1)


# In[22]:


grid = GridSearchCV(SVC(),param_grid,cv=5)
grid.fit(data,labels)


# In[26]:


joblib.dump(grid, 'grid.pkl')


# In[29]:


Y_test=test_data[b'labels']
X_test=test_data[b'data']


# In[37]:


X_test = joblib.load('Xtest.pkl')


# In[ ]:


# train the model on train set 
model = SVC() 
model.fit(data, labels) 
  
# print prediction results of train
predictions = model.predict(data) 
print(classification_report(labels, predictions)) 
print(accuracy_score(labels, predictions))

# print prediction results of train
predictions = model.predict(X_test) 
print(classification_report(Y_test, predictions)) 
print(accuracy_score(Y_test, predictions))


# In[ ]:




