#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
df = pd.read_csv('ccdefault.csv')
df = df[df.columns[1:]]


# In[41]:


#import sklearn
from sklearn.model_selection import train_test_split
X = df.iloc[:, 0:23].values
y=df[df.columns[23]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)


# In[3]:


from sklearn.ensemble import RandomForestClassifier


# In[10]:


#
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {'n_estimators': [10,20,50,100,125,150,200,300]}
# Create a based model
rf = RandomForestClassifier(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 10, n_jobs = -1, verbose = 1, return_train_score=True)


# In[11]:


grid_search


# In[12]:


grid_search.fit(X_train, y_train)


# In[13]:


mean_train_score =grid_search.cv_results_['mean_train_score']
print(mean_train_score)


# In[14]:


mean_test_score =grid_search.cv_results_['mean_test_score']
print(mean_test_score)


# In[15]:


grid_search.best_params_


# In[19]:


#build the randomforestclassifier model with the best parameters
rfc = RandomForestClassifier(n_estimators=150,random_state=42)
#fit the model with training set
rfc.fit(X_train,y_train)


# In[20]:


# Get feature importances from our random forest model

importances = rfc.feature_importances_


# In[43]:


importances


# In[44]:


# Get the index of importances from greatest importance to least
import numpy as np
sorted_index = np.argsort(importances)[::-1]

x = range(len(importances))


# In[45]:


sorted_index


# In[48]:


dX = df.iloc[:, 0:23]
df.feature_names = list(dX.columns.values) 
df.class_names = df.columns[23]


# In[53]:


labels =np.array(df.feature_names)[sorted_index]


# In[57]:


import matplotlib.pyplot as plt
plt.bar(x, importances[sorted_index], tick_label=labels)
# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()


# In[58]:


print("My name is Xuehui Chao")
print("My NetID is: xuehuic2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

