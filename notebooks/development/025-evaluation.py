#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import json

import pandas as pd

from smote_variants import get_simplex_sampling_oversamplers
from common_datasets.binary_classification import get_filtered_data_loaders

from smote_variants.evaluation import evaluate_oversamplers

import smote_variants
smote_variants.config.suppress_external_warnings(False)
smote_variants.config.suppress_internal_warnings(False)

logger = logging.getLogger('smote_variants')
logger.setLevel(logging.CRITICAL)


# In[2]:


oversamplers = get_simplex_sampling_oversamplers(within_simplex_sampling='random', 
                                                    n_dim_range=2, 
                                                    n_quickest=50)
oversamplers = [smote_variants.NEATER]


# In[3]:


datasets = get_filtered_data_loaders(n_col_bounds=(2, 150), 
                                        n_minority_bounds=(10, 10000), 
                                        n_bounds=(50, 2500),
                                        n_smallest=100, 
                                        sorting='n')

datasets = [dataset() for dataset in datasets]

datasets = [dataset for dataset in datasets if 'kddcup' in dataset['name']]


# In[4]:


len(datasets)


# In[5]:


classifiers = [('sklearn.neighbors', 'KNeighborsClassifier', {'algorithm': 'brute', 
                                                                'n_jobs': 1}),
                ('sklearn.tree', 'DecisionTreeClassifier', {'random_state': 5})]


# In[6]:


validator_params = {'n_repeats': 1, 'n_splits': 2, 'random_state': 5}

ss_params = {'within_simplex_sampling': 'deterministic',
             'simplex_sampling': 'deterministic'}

vanilla_params = {'random_state': 5, 'n_jobs': 1}
deterministic_params = {'random_state': 5,
                        'ss_params': ss_params,
                        'n_jobs': 1}


# In[7]:


cache_path='/home/gykovacs/smote-deterministic2/'


# In[8]:


# creating oversampler specifications

oversampler_list = [('smote_variants', o.__name__, vanilla_params) for o in oversamplers]
oversampler_deterministic = [('smote_variants', o.__name__, deterministic_params) for o in oversamplers]

all_oversamplers = oversampler_list + oversampler_deterministic

#all_oversamplers = [('smote_variants', 'TRIM_SMOTE', {'random_state': 5})]


# In[9]:


print(len(all_oversamplers))
print(len(datasets))


# In[10]:


results = evaluate_oversamplers(datasets=datasets, 
                                oversamplers=all_oversamplers, 
                                classifiers=[('sklearn.neighbors', 'KNeighborsClassifier', {})],
                                scaler=('sklearn.preprocessing', 'StandardScaler', {}),
                                validator_params={'n_repeats': 1, 'n_splits': 2, 'random_state': 5},
                                cache_path=cache_path,
                                parse_results=False,
                                n_jobs=1,
                                clean_up=None)


# In[11]:


print(len(results))


# In[12]:


def load_json(path):
    with open(path, 'rt') as file:
        return json.load(file)


# In[13]:


data = [load_json(path) for path in results]


# In[14]:


pdf = pd.DataFrame(data)


# In[15]:


pdf


# In[16]:


import numpy as np


# In[17]:


def counts_to_vector(counts):
    """
    Expand a count vector to a 
    
    Args:
        counts (np.array): count vector
    
    Returns:
        np.array: the expanded vector
    """
    
    return np.hstack([np.repeat(idx, count) for idx, count in enumerate(counts)])


# In[ ]:





# In[18]:


def counts_to_vector(counts):
    """
    Expand a count vector to a 
    
    Args:
        counts (np.array): count vector
    
    Returns:
        np.array: the expanded vector
    """
    
    return np.hstack([np.repeat(idx, count) for idx, count in enumerate(counts)])

def deterministic_sample(choices, n_to_sample, p):
    """
    Take a deterministic sample
    
    Args:
        choices (list): the list of choices
        n_to_sample (int): the number of samples to take
        p (np.array): the distribution
    
    Returns:
        np.array: the choices
    """
    
    sample_counts = np.ceil(n_to_sample * p).astype(int)
    
    n_to_remove = np.sum(sample_counts) - n_to_sample
    
    if n_to_remove == 0:
        return choices[counts_to_vector(sample_counts)]
    
    non_zero_mask = sample_counts > 0

    removal_indices = np.floor(np.linspace(0.0, np.sum(non_zero_mask), n_to_remove, endpoint=False)).astype(int)

    tmp = sample_counts[non_zero_mask]
    tmp[removal_indices] = tmp[removal_indices] - 1
    
    sample_counts[non_zero_mask] = tmp

    assert np.sum(sample_counts) == n_to_sample
    
    samples = choices[counts_to_vector(sample_counts)]
    
    return samples
    


# In[19]:


tmp = pd.DataFrame({'name': ['a', 'b', 'a', 'c', 'a'], 'value': [0, 1, 2, 3, 4]})


# In[20]:


tmp.groupby('name').head(1)


# In[21]:



deterministic_sample(np.arange(1), 3, np.array([1]))


# In[ ]:





# In[22]:


import numpy as np


# In[23]:


tmp = np.array([1, 2, 3, 4, 5])


# In[24]:


tmp[::2] = tmp[::2] - 1


# In[25]:


tmp


# In[ ]:




