
# coding: utf-8

# # Evaluation of oversamplers with a set of classifiers on a set of datasets
# 
# In this notebook, we give an example of evaluating multiple oversamplers on multiple datasets with multiple classifiers. 

# In[1]:


import os.path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import smote_variants as sv

import imbalanced_databases as imbd


# In[2]:


# Setting the cache_path which is used for caching during the evaluation

cache_path= os.path.join(os.path.expanduser('~'), 'smote_test')

if not os.path.exists(cache_path):
    os.makedirs(cache_path)


# In[3]:


# Specifying two datasets by their load functions

datasets= [imbd.load_glass0, imbd.load_yeast1]


# In[4]:


# Specifying the classifiers used for evaluation

knn_classifier= KNeighborsClassifier()
dt_classifier= DecisionTreeClassifier()


# In[5]:


# Executing the evaluation using 5 parallel jobs, and at most 35 different 
# random but meaningful parameter combinations with the oversamplers

results= sv.evaluate_oversamplers(datasets= datasets,
                                    samplers= sv.get_n_quickest_oversamplers(5),
                                    classifiers= [knn_classifier, dt_classifier],
                                    cache_path= cache_path,
                                    n_jobs= 5,
                                    max_samp_par_comb= 35)


# In[6]:


# The results are arranged in a pandas DataFrame with the following columns:
# db_name - name of the database
# classifier - name of the classifier
# sampler - name of the oversampling technique
# auc - highest auc score with the classifier and oversampler (aggregated over all classifier and oversampler
# parameter combinations)
# brier - highest brier score with the classifier and oversampler (aggregated similarly)
# acc - the highest accuracy score with the classifier and oversampler (aggregated similarly)
# f1 - the highest f1 score with the classifier and oversampler (aggregated similarly)
# p_top20 - the highest p_top20 score with the classifier and oversampler (aggregated similarly)
# gacc - the highest GACC score with the classifier and oversampler (aggregated similarly)
# runtime - average runtime in seconds
# db_size - size of the dataset
# db_n_attr - number of attributes in the dataset
# imbalanced_ratio - the ratio of majority/minority class sizes
# sampler_categories - the categories assigned to the oversampler
# classifier_parameters_auc - the classifier parameters reaching the highest auc score
# classifier_parameters_acc - the classifier parameters reaching the highest acc score
# classifier_parameters_gacc - the classifier parameters reaching the highest gacc score
# classifier_parameters_f1 - the classifier parameters reaching the highest f1 score
# classifier_parameters_p_top20 - the classifier parameters reaching the highest p_top20 score
# classifier_parameters_brier - the classifier parameters reaching the highest brier score
# sampler_parameters_auc - the oversampler parameters reaching the highest auc score
# sampler_parameters_acc - the oversampler parameters reaching the highest acc score
# sampler_parameters_gacc - the oversampler parameters reaching the highest gacc score
# sampler_parameters_f1 - the oversampler parameters reaching the highest f1 score
# sampler_parameters_p_top20 - the oversampler parameters reaching the highest p_top20 score
# sampler_parameters_brier - the oversampler parameters reaching the highest brier score

print(results.columns)


# In[7]:


# The results can be processed according to the requirements of the analysis

print(results)

