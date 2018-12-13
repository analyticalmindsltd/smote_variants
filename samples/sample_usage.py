#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs

This script illustrates the use of oversampling techniques with various
parameterizations
"""

import smote_variants as sv
import sklearn.datasets as datasets

dataset= datasets.load_wine()

oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())

X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
