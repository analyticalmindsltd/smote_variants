#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs

This script illustrates the use of oversampling techniques with various
parameterizations
"""

import numpy as np

# datasets
from sklearn import datasets

# import SMOTE variants
import smote_variants as sv

# prepare dataset
dataset= datasets.load_breast_cancer()
X, y= dataset['data'], dataset['target']

################
# Oversampling #
################

oversampler= sv.Borderline_SMOTE1()
X_samp, y_samp= oversampler.sample(X, y)
print('original length: %d oversampled length: %d' % (len(X), len(X_samp)))

#############################################
# Oversampling with user defined parameters #
#############################################

oversampler= sv.SMOTE_ENN(proportion= 0.5, n_neighbors= 3)
X_samp, y_samp= oversampler.sample(X, y)
print('original length: %d oversampled length: %d' % (len(X), len(X_samp)))

##################################################
# Oversampling with random reasonable parameters #
##################################################

oversampler= sv.ADASYN(**np.random.choice(sv.ADASYN.parameter_combinations()))
X_samp, y_samp= oversampler.sample(X, y)
print('original length: %d oversampled length: %d' % (len(X), len(X_samp)))
