#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:


import matplotlib.pyplot as plot
import os


# In[3]:


working_directory = os.getcwd()

print(working_directory)


# In[4]:


path = '/Users/pradeepvallepalli/Downloads/Iris.csv'
df = pd.read_csv(path,names = ['f1','f2','f3','f4','c'])
import csv

with open('/Users/pradeepvallepalli/Downloads/Iris.csv') as handle:
    reader = csv.reader(handle, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)

with open('/Users/pradeepvallepalli/Downloads/Iris.csv', 'r') as iris_data:
    irises = list(csv.reader(iris_data))
df


# In[5]:


df.info()


# In[6]:


df.head()


# In[11]:


from sklearn.model_selection import train_test_split
labels
features


# In[12]:


features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(
features, labels, test_size=0.1, random_state=100
)


# In[14]:


from sklearn.model_selection import train_test_split
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[15]:



from sklearn import svm

svc = svm.SVC(kernel='linear',C=1,gamma=1)
svc.fit(features,labels)
y_pred = svc.predict(x_test)

# clf_svm_l=svm.SVC(kernel='linear',C=1,gamma=1)

# clf_svm_l.fit(features,labels)
# y_pred = clf_svm_l.predict(x_test)
# knn = KNeighborsClassifier()
# knn.fit(x_train,y_train)


# In[ ]:


y_pred.shape


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# In[ ]:




