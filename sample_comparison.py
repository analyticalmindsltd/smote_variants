#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs

Sample script on how to evaluate various classifiers and oversampling techniques
on various datasets.
"""

import os.path

# import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# import SMOTE variants
import smote_variants as sv

# import datasets
import imbalanced_databases as imbd

# setting cache path
cache_path= os.path.join(os.path.expanduser('~'), 'workspaces', 'smote_test')

# instantiating classifiers
knn_classifier= KNeighborsClassifier()
dt_classifier= DecisionTreeClassifier()

# instantiate the validation object
results= sv.evaluate_oversamplers(datasets= imbd.get_data_loaders('tiny'),
                                  samplers= sv.get_n_quickest_oversamplers(5),
                                  classifiers= [knn_classifier, dt_classifier],
                                  cache_path= cache_path)

# results of the evaluation
print(results)
