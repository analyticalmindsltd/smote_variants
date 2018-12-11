#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs

A simplified sample script on how to evaluate various classifiers and oversampling techniques
on a given dataset to find the best performing one.
"""

import os.path

# import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# import SMOTE variants
import smote_variants as sv

# import datasets
from sklearn import datasets

# setting cache path
cache_path= os.path.join(os.path.expanduser('~'), 'workspaces', 'smote_test')

# prepare dataset
dataset= datasets.load_breast_cancer()
dataset= {'data': dataset['data'], 'target': dataset['target'], 'name': 'breast_cancer'}

# instantiating classifiers
knn_classifier= KNeighborsClassifier()
dt_classifier= DecisionTreeClassifier()

# instantiate the validation object
samp_obj, cl_obj= sv.model_selection(datasets= [dataset],
                                      samplers= sv.get_n_quickest_oversamplers(5),
                                      classifiers= [knn_classifier, dt_classifier],
                                      cache_path= cache_path,
                                      n_jobs= 5,
                                      max_n_sampler_parameters= 35)

# oversampling and classifier training
X_samp, y_samp= samp_obj.sample(dataset['data'], dataset['target'])
cl_obj.fit(X_samp, y_samp)
