
# coding: utf-8

# # Integration with sklearn pipelines
# 
# In this notebook, provide some illustration for integration with sklearn pipelines.

# In[1]:


import keras
import imblearn

import numpy as np

import smote_variants as sv
import imblearn.datasets as imb_datasets

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

random_seed= 3


# ## Preparing the data

# In[2]:


np.random.seed(random_seed)


# In[3]:


libras= imb_datasets.fetch_datasets()['libras_move']
X, y= libras['data'], libras['target']


# In[4]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.33)


# ## Fitting a pipeline

# In[5]:


oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())
classifier= KNeighborsClassifier(n_neighbors= 5)


# In[6]:


model= Pipeline([('scale', StandardScaler()), ('clf', sv.OversamplingClassifier(oversampler, classifier))])


# In[7]:


model.fit(X, y)


# ## Grid search

# In[8]:


param_grid= {'clf__oversampler':[sv.distance_SMOTE(proportion=0.5),
                                 sv.distance_SMOTE(proportion=1.0),
                                 sv.distance_SMOTE(proportion=1.5)]}


# In[9]:


grid= GridSearchCV(model, param_grid= param_grid, cv= 3, n_jobs= 1, verbose= 2, scoring= 'accuracy')


# In[10]:


grid.fit(X, y)


# In[11]:


print(grid.best_score_)
print(grid.cv_results_)

