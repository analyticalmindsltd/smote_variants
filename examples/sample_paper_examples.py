#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:27:44 2018

@author: gykovacs
"""

#%% importing necessary packages

import logging
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

#%% setting output format for pandas

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 10)
pd.set_option('max_colwidth', 1000)
pd.set_option('expand_frame_repr', False)

#logging.getLogger('smote_variants').setLevel(logging.ERROR)

#%% print package versions

print('numpy %s' % np.__version__)
print('imblearn %s' % imblearn.__version__)
print('scipy %s' % scipy.__version__)
print('sklearn %s' % sklearn.__version__)
print('keras %s' % keras.__version__)
print('smote_variants %s' % sv.__version__)

#%% defining plotting functions

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

#%% reading datasets
    
datasets= imb_datasets.fetch_datasets()

libras= datasets['libras_move']
ecoli= datasets['ecoli']
wine= sk_datasets.load_wine()

#%% running OUPS

np.random.seed(random_seed)
oups= sv.OUPS()
X, y= oups.sample(libras['data'], libras['target'])

plot(libras['data'], libras['target'], 'libras_move', 1, -1, 'libras_move.eps')
plot(X, y, 'libras_move oversampled by OUPS', 1, -1, 'libras_move_oups.eps')

#%% cross-validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

np.random.seed(random_seed)
results= sv.cross_validate(dataset= libras, 
                           sampler= sv.OUPS(), 
                           classifier= KNeighborsClassifier(),
                           validator= RepeatedStratifiedKFold(n_repeats= 8,
                                                              n_splits= 5))
print(results.T[['sampler', 'auc', 'gacc']])

np.random.seed(random_seed)
results= sv.cross_validate(dataset= libras, 
                           sampler= sv.NoSMOTE(), 
                           classifier= KNeighborsClassifier(),
                           validator= RepeatedStratifiedKFold(n_repeats= 8,
                                                              n_splits= 5))
print(results.T[['sampler', 'auc', 'gacc']])

#%% running multiclass oversampling

np.random.seed(random_seed)
mc_oversampler= sv.MulticlassOversampling(sv.distance_SMOTE(), strategy= 'equalize_1_vs_many_successive')
X_os, y_os= mc_oversampler.sample(wine['data'], wine['target'])

plot_mc(wine['data'], wine['target'], 'wine', 0, 1, 2, 'wine.eps')
plot_mc(X_os, y_os, 'wine oversampled by distance-SMOTE', 0, 1, 2, 'wine_distance_smote.eps')

#%% oversampler evaluation

import os.path

ecoli['name']= 'ecoli'
cache_path= os.path.join(os.path.expanduser('~'), 'smote_cache')

np.random.seed(random_seed)
results= sv.evaluate_oversamplers(datasets= [ecoli], 
                                  samplers= [sv.SPY,
                                             sv.OUPS,
                                             sv.NoSMOTE], 
                                  classifiers= [KNeighborsClassifier()],
                                  validator= RepeatedStratifiedKFold(n_repeats= 3,
                                                                     n_splits= 5),
                                  cache_path= cache_path,
                                  max_samp_par_comb= 3,
                                  all_results= True,
                                  n_jobs= 6)

print(results[['sampler', 'sampler_parameters', 'auc']])

#%% oversampler selection

from sklearn.tree import DecisionTreeClassifier

np.random.seed(random_seed)
samp, clas= sv.model_selection(dataset= ecoli, 
                               samplers= sv.get_all_oversamplers(), 
                               classifiers= [KNeighborsClassifier(),
                                             DecisionTreeClassifier()],
                               validator= RepeatedStratifiedKFold(n_repeats= 3,
                                                                  n_splits= 5),
                               score= 'auc',
                               cache_path= cache_path,
                               max_samp_par_comb= 3,
                               n_jobs= 6)

print(samp)
print(clas)
