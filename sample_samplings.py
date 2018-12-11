#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:37:20 2018

@author: gykovacs

Sample script on how to evaluate various classifiers and oversampling techniques
on a given dataset to find the best performing one. Also, illustration on
the processing of the results.
"""

import os.path

# import classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# import SMOTE variants
import smote_variants as sv

# import datasets
from sklearn import datasets

oversamplers= sv.get_all_oversamplers()

for o in oversamplers:
    print(o.__name__)
    sv.ballpark_sample(o(), img_file_base= 'base.png', img_file_sampled= ('%s.png' % (o.__name__)))
