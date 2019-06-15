
# coding: utf-8

# # Evaluation of oversamplers with a set of classifiers on one database
# 
# In this notebook we give an example of optimizing oversamplers and classifiers for given dataset.

# In[1]:


import os.path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import smote_variants as sv

import imbalanced_databases as imbd


# In[2]:


# the evaluation procedure uses a directory for caching

cache_path= os.path.join(os.path.expanduser('~'), 'smote_test')

if not os.path.exists(cache_path):
    os.makedirs(cache_path)


# In[3]:


# specifying the dataset to be used

dataset= imbd.load_glass0()


# In[4]:


# specifying the classifiers

knn_classifier= KNeighborsClassifier()
dt_classifier= DecisionTreeClassifier()


# In[5]:


# executing the evaluation using 5 parallel jobs and at most 35 random but meaningful parameter combinations
# with the 5 quickest oversamplers

results= sv.evaluate_oversamplers(datasets= [dataset],
                                    samplers= sv.get_n_quickest_oversamplers(5),
                                    classifiers= [knn_classifier, dt_classifier],
                                    cache_path= cache_path,
                                    n_jobs= 5,
                                    max_samp_par_comb= 35)


# In[6]:


# determining oversampler and classifier combination with highest AUC score

highest_auc_score= results['auc'].idxmax()


# In[7]:


# querying classifier and oversampler parameters with highest AUC score

cl, cl_par, samp, samp_par= results.loc[highest_auc_score][['classifier',
                                                           'classifier_parameters_auc',
                                                           'sampler',
                                                           'sampler_parameters_auc']]


# In[8]:


# instantiating oversampler and classifier objects providing the highest AUC score

samp_obj= getattr(sv, samp)(**eval(samp_par))
cl_obj= eval(cl)(**eval(cl_par))


# In[9]:


# oversampling the entire dataset and fitting a classifier

X_samp, y_samp= samp_obj.sample(dataset['data'], dataset['target'])
cl_obj.fit(X_samp, y_samp)

