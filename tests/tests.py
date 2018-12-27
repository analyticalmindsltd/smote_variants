#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 19:21:49 2018

@author: gykovacs
"""

import numpy as np

import logging

import smote_variants as sv

def validation(smote, X, y):
    return smote.sample(X, y)

def test_some_min_some_maj():
    X= np.array([[1.0, 1.1],
        [1.1, 1.2],
        [1.05, 1.1],
        [1.08, 1.05],
        [1.1, 1.08],
        [1.5, 1.6],
        [1.55, 1.55]])
        
    y= np.array([0, 0, 0, 0, 0, 1, 1])
    
    samplers= sv.get_all_oversamplers()
    
    for s in samplers:
        logging.info("testing %s" % str(s))
        X_samp, y_samp= validation(s(), X, y)
        assert len(X_samp) > 0

def test_1_min_some_maj():
    X= np.array([[1.0, 1.1],
        [1.1, 1.2],
        [1.05, 1.1],
        [1.08, 1.05],
        [1.1, 1.08],
        [1.55, 1.55]])
        
    y= np.array([0, 0, 0, 0, 0, 1])
    
    samplers= sv.get_all_oversamplers()
    
    for s in samplers:
        logging.info("testing %s" % str(s))
        X_samp, y_samp= validation(s(), X, y)
        assert len(X_samp) > 0

def test_1_min_1_maj():
    X= np.array([[1.0, 1.1],
        [1.55, 1.55]])
        
    y= np.array([0, 1])
    
    samplers= sv.get_all_oversamplers()
    
    for s in samplers:
        logging.info("testing %s" % str(s))
        X_samp, y_samp= validation(s(), X, y)
        assert len(X_samp) > 0
