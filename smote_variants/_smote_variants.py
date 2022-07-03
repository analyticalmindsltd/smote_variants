#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 11:15:24 2018

@author: gykovacs
"""

# import system packages
import os
import pickle
import itertools
import logging
import re
import time
import glob
import inspect

# used to parallelize evaluation
from joblib import Parallel, delayed

# numerical methods and arrays
import numpy as np
import pandas as pd

# import packages used for the implementation of sampling methods
from sklearn.model_selection import (RepeatedStratifiedKFold, KFold,
                                     cross_val_score, StratifiedKFold)
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone, BaseEstimator, ClassifierMixin

# some statistical methods
from scipy.stats import skew
import scipy.signal as ssignal
import scipy.spatial as sspatial
import scipy.optimize as soptimize
import scipy.special as sspecial
from scipy.stats.mstats import gmean
from sklearn.cluster import OPTICS

from ._version import __version__
from ._base import *
from ._logger import logger
_logger = logger

from .oversampling._OverSampling import OverSampling
from .oversampling._SMOTE import SMOTE
from .oversampling._ADASYN import ADASYN
from .oversampling._polynom_fit_SMOTE import polynom_fit_SMOTE
from .noise_removal import NeighborhoodCleaningRule

__author__ = "György Kovács"
__license__ = "MIT"
__email__ = "gyuriofkovacs@gmail.com"

# exported names
__all__ = ['__author__',
           '__license__',
           '__version__',
           '__email__',
           'MulticlassOversampling']

class MulticlassOversampling(StatisticsMixin):
    """
    Carries out multiclass oversampling

    Example::

        import smote_variants as sv
        import sklearn.datasets as datasets

        dataset= datasets.load_wine()

        oversampler= sv.MulticlassOversampling(sv.distance_SMOTE())

        X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
    """

    def __init__(self,
                 oversampler=SMOTE(random_state=2),
                 strategy="eq_1_vs_many_successive"):
        """
        Constructor of the multiclass oversampling object

        Args:
            oversampler (obj): an oversampling object
            strategy (str/obj): a multiclass oversampling strategy, currently
                                'eq_1_vs_many_successive' or
                                'equalize_1_vs_many'
        """
        self.oversampler = oversampler
        self.strategy = strategy

    def sample_equalize_1_vs_many(self, X, y):
        """
        Does the sample generation by oversampling each minority class to the
        cardinality of the majority class using all original samples in each
        run.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        message = "Running multiclass oversampling with strategy %s"
        message = message % str(self.strategy)
        _logger.info(self.__class__.__name__ + ": " + message)

        if 'proportion' not in self.oversampler.get_params():
            message = ("Multiclass oversampling strategy %s cannot be "
                       "used with oversampling techniques without proportion"
                       " parameter")
            message = message % str(self.strategy)
            raise ValueError(message)

        # extract class label statistics
        self.class_label_statistics(X, y)

        # sort labels by number of samples
        class_labels = self.class_stats.keys()
        class_labels = sorted(class_labels, key=lambda x: -self.class_stats[x])

        majority_class_label = class_labels[0]

        # determining the majority class data
        X_maj = X[y == majority_class_label]

        # dict to store the results
        results = {}
        results[majority_class_label] = X_maj.copy()

        # running oversampling for all minority classes against all oversampled
        # classes
        for i in range(1, len(class_labels)):
            message = "Sampling minority class with label: %d"
            message = message % class_labels[i]
            _logger.info(self.__class__.__name__ + ": " + message)

            # extract current minority class
            minority_class_label = class_labels[i]
            X_min = X[y == minority_class_label]
            X_maj = X[y != minority_class_label]

            # prepare data to pass to oversampling
            X_training = np.vstack([X_maj, X_min])
            y_training = np.hstack(
                [np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler.get_params()

            num_to_generate = self.class_stats[majority_class_label] - \
                self.class_stats[class_labels[i]]
            num_to_gen_to_all = len(X_maj) - self.class_stats[class_labels[i]]

            params['proportion'] = num_to_generate/num_to_gen_to_all

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = self.oversampler.__class__(**params)

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_training, y_training)

            # registaring the newly oversampled minority class in the output
            # set
            results[class_labels[i]] = X_samp[len(
                X_training):][y_samp[len(X_training):] == 1]

        # constructing the output set
        X_final = results[class_labels[1]]
        y_final = np.repeat(class_labels[1], len(results[class_labels[1]]))

        for i in range(2, len(class_labels)):
            X_final = np.vstack([X_final, results[class_labels[i]]])
            y_new = np.repeat(class_labels[i], len(results[class_labels[i]]))
            y_final = np.hstack([y_final, y_new])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample_equalize_1_vs_many_successive(self, X, y):
        """
        Does the sample generation by oversampling each minority class
        successively to the cardinality of the majority class,
        incorporating the results of previous oversamplings.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        message = "Running multiclass oversampling with strategy %s"
        message = message % str(self.strategy)
        _logger.info(self.__class__.__name__ + ": " + message)

        if 'proportion' not in self.oversampler.get_params():
            message = ("Multiclass oversampling strategy %s cannot be used"
                       " with oversampling techniques without proportion"
                       " parameter") % str(self.strategy)
            raise ValueError(message)

        # extract class label statistics
        self.class_label_statistics(X, y)

        # sort labels by number of samples
        class_labels = self.class_stats.keys()
        class_labels = sorted(class_labels, key=lambda x: -self.class_stats[x])

        majority_class_label = class_labels[0]

        # determining the majority class data
        X_maj = X[y == majority_class_label]

        # dict to store the results
        results = {}
        results[majority_class_label] = X_maj.copy()

        # running oversampling for all minority classes against all
        # oversampled classes
        for i in range(1, len(class_labels)):
            message = "Sampling minority class with label: %d"
            message = message % class_labels[i]
            _logger.info(self.__class__.__name__ + ": " + message)

            # extract current minority class
            minority_class_label = class_labels[i]
            X_min = X[y == minority_class_label]

            # prepare data to pass to oversampling
            X_training = np.vstack([X_maj, X_min])
            y_training = np.hstack(
                [np.repeat(0, len(X_maj)), np.repeat(1, len(X_min))])

            # prepare parameters by properly setting the proportion value
            params = self.oversampler.get_params()

            n_majority = self.class_stats[majority_class_label]
            n_class_i = self.class_stats[class_labels[i]]
            num_to_generate = n_majority - n_class_i

            num_to_gen_to_all = i * n_majority - n_class_i

            params['proportion'] = num_to_generate/num_to_gen_to_all

            # instantiating new oversampling object with the proper proportion
            # parameter
            oversampler = self.oversampler.__class__(**params)

            # executing the sampling
            X_samp, y_samp = oversampler.sample(X_training, y_training)

            # adding the newly oversampled minority class to the majority data
            X_maj = np.vstack([X_maj, X_samp[y_samp == 1]])

            # registaring the newly oversampled minority class in the output
            # set
            result_mask = y_samp[len(X_training):] == 1
            results[class_labels[i]] = X_samp[len(X_training):][result_mask]

        # constructing the output set
        X_final = results[class_labels[1]]
        y_final = np.repeat(class_labels[1], len(results[class_labels[1]]))

        for i in range(2, len(class_labels)):
            X_final = np.vstack([X_final, results[class_labels[i]]])
            y_new = np.repeat(class_labels[i], len(results[class_labels[i]]))
            y_final = np.hstack([y_final, y_new])

        return np.vstack([X, X_final]), np.hstack([y, y_final])

    def sample(self, X, y):
        """
        Does the sample generation according to the oversampling strategy.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        if self.strategy == "eq_1_vs_many_successive":
            return self.sample_equalize_1_vs_many_successive(X, y)
        elif self.strategy == "equalize_1_vs_many":
            return self.sample_equalize_1_vs_many(X, y)
        else:
            message = "Multiclass oversampling startegy %s not implemented."
            message = message % self.strategy
            raise ValueError(message)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the multiclass oversampling object
        """
        return {'oversampler': self.oversampler, 'strategy': self.strategy}


