#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs

This script contains the code used in the study on the evaluation of oversampling
techniques.
"""

import os, pickle, itertools

# import classifiers
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from smote_variants import MLPClassifierWrapper

# import SMOTE variants
import smote_variants as sv

# itertools to derive imbalanced databases
import imbalanced_databases as imbd

# global variables
cache_path= '/home/gykovacs/workspaces/smote_results/'
max_sampler_parameter_combinations= 35
n_jobs= 35

# instantiate classifiers
sv_classifiers= [CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=1.0, penalty='l2', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l1', loss= 'squared_hinge', dual= False)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'hinge', dual= True)),
                 CalibratedClassifierCV(LinearSVC(C=10.0, penalty='l2', loss= 'squared_hinge', dual= False))]

mlp_classifiers= []
for x in itertools.product(['relu', 'logistic'], [1.0, 0.5, 0.1]):
    mlp_classifiers.append(MLPClassifierWrapper(activation= x[0], hidden_layer_fraction= x[1]))

nn_classifiers= []
for x in itertools.product([3, 5, 7], ['uniform', 'distance'], [1, 2, 3]):
    nn_classifiers.append(KNeighborsClassifier(n_neighbors= x[0], weights= x[1], p= x[2]))

dt_classifiers= []
for x in itertools.product(['gini', 'entropy'], [None, 3, 5]):
    dt_classifiers.append(DecisionTreeClassifier(criterion= x[0], max_depth= x[1]))

classifiers= []
classifiers.extend(sv_classifiers)
classifiers.extend(mlp_classifiers)
classifiers.extend(nn_classifiers)
classifiers.extend(dt_classifiers)

samplers= [sv.SMOTE, sv.SMOTE_TomekLinks, sv.SMOTE_ENN, sv.MSYN, sv.SVM_balance,
           sv.SMOTE_RSB, sv.NEATER, sv.DEAGO, sv.SMOTE_IPF, sv.ISOMAP_Hybrid,
           sv.E_SMOTE, sv.SMOTE_PSOBAT, sv.SMOTE_FRST_2T, sv.AMSCO, sv.NDO_sampling,
           sv.DSRBF]

results= sv.evaluate_oversamplers(datasets= imbd.get_data_loaders('study'),
                                  samplers= samplers,
                                  classifiers= classifiers,
                                  cache_path= cache_path,
                                  n_jobs= n_jobs)

pickle.dump(results, open(os.path.join(cache_path, 'results.pickle'), 'wb'))