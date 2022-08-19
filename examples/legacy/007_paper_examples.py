
# coding: utf-8

# # Examples from the paper
# 
# In this notebook, provide the codes used for illustration in the corresponding paper with all the supplementary code segments excluded from the paper due to space limitations.

# In[1]:


import scipy
import sklearn
import keras
import imblearn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import smote_variants as sv
import imblearn.datasets as imb_datasets
import sklearn.datasets as sk_datasets

random_seed= 3


# In[2]:


# configuring pandas to print all columns

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 10)
pd.set_option('max_colwidth', 1000)
pd.set_option('expand_frame_repr', False)


# In[3]:


# printing package versions

print('numpy %s' % np.__version__)
print('imblearn %s' % imblearn.__version__)
print('scipy %s' % scipy.__version__)
print('sklearn %s' % sklearn.__version__)
print('keras %s' % keras.__version__)
print('smote_variants %s' % sv.__version__)


# In[4]:


# defining some plotting functions

def plot(X, y, title, min_label, maj_label, filename):
    plt.figure(figsize= (4, 3))
    plt.scatter(X[:,0][y == min_label], X[:,1][y == min_label], label='minority class', color='red', s=25)
    plt.scatter(X[:,0][y == maj_label], X[:,1][y == maj_label], label='majority class', color='black', marker='*', s=25)
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
def plot_mc(X, y, title, label_0, label_1, label_2, filename):
    plt.figure(figsize= (4, 3))
    plt.scatter(X[:,0][y == label_0], X[:,1][y == label_0], label='class 0', color='red', s=25)
    plt.scatter(X[:,0][y == label_1], X[:,1][y == label_1], label='class 1', color='black', marker='*', s=25)
    plt.scatter(X[:,0][y == label_2], X[:,1][y == label_2], label='class 2', color='blue', marker='^', s=25)
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# In[5]:


# setting the random seed
np.random.seed(random_seed)


# In[6]:


# sample code segment #0
# oversampling by OUPS and plotting

import smote_variants as sv
import imblearn.datasets as imb_datasets

libras= imb_datasets.fetch_datasets()['libras_move']

X, y= sv.OUPS().sample(libras['data'], libras['target'])

plot(libras['data'], libras['target'], 'libras_move', 1, -1, 'libras_move.eps')
plot(X, y, 'libras_move oversampled by OUPS', 1, -1, 'libras_move_oups.eps')


# In[7]:


# setting the random seed
np.random.seed(random_seed)


# In[8]:


# sample code segment #1
# evaluating the performance of k neighbors classifier with oversampling

from sklearn.neighbors import KNeighborsClassifier

results= sv.cross_validate(dataset= libras, sampler= sv.OUPS(), 
                           classifier= KNeighborsClassifier())

print(results.loc['auc'])


# In[9]:


# evaluating the performance of k neighbors classifier without oversampling

np.random.seed(random_seed)
results_wo= sv.cross_validate(dataset= libras, sampler= sv.NoSMOTE(), 
                               classifier= KNeighborsClassifier())


# In[10]:


# printing the results

print(results_wo.loc['auc'])

