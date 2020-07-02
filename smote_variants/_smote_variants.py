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

# self-organizing map implementation
import minisom

from ._version import __version__

__author__ = "György Kovács"
__license__ = "MIT"
__email__ = "gyuriofkovacs@gmail.com"

# for handler in _logger.root.handlers[:]:
#    _logger.root.removeHandler(handler)

# setting the _logger format
_logger = logging.getLogger('smote_variants')
_logger.setLevel(logging.DEBUG)
_logger_ch = logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter(
    "%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)

# exported names
__all__ = ['__author__',
           '__license__',
           '__version__',
           '__email__',
           'get_all_oversamplers',
           'get_all_noisefilters',
           'get_n_quickest_oversamplers',
           'get_all_oversamplers_multiclass',
           'get_n_quickest_oversamplers_multiclass',
           'evaluate_oversamplers',
           'read_oversampling_results',
           'model_selection',
           'cross_validate',
           'MLPClassifierWrapper',
           'OverSampling',
           'NoiseFilter',
           'TomekLinkRemoval',
           'CondensedNearestNeighbors',
           'OneSidedSelection',
           'CNNTomekLinks',
           'NeighborhoodCleaningRule',
           'EditedNearestNeighbors',
           'SMOTE',
           'SMOTE_TomekLinks',
           'SMOTE_ENN',
           'Borderline_SMOTE1',
           'Borderline_SMOTE2',
           'ADASYN',
           'AHC',
           'LLE_SMOTE',
           'distance_SMOTE',
           'SMMO',
           'polynom_fit_SMOTE',
           'Stefanowski',
           'ADOMS',
           'Safe_Level_SMOTE',
           'MSMOTE',
           'DE_oversampling',
           'SMOBD',
           'SUNDO',
           'MSYN',
           'SVM_balance',
           'TRIM_SMOTE',
           'SMOTE_RSB',
           'ProWSyn',
           'SL_graph_SMOTE',
           'NRSBoundary_SMOTE',
           'LVQ_SMOTE',
           'SOI_CJ',
           'ROSE',
           'SMOTE_OUT',
           'SMOTE_Cosine',
           'Selected_SMOTE',
           'LN_SMOTE',
           'MWMOTE',
           'PDFOS',
           'IPADE_ID',
           'RWO_sampling',
           'NEATER',
           'DEAGO',
           'Gazzah',
           'MCT',
           'ADG',
           'SMOTE_IPF',
           'KernelADASYN',
           'MOT2LD',
           'V_SYNTH',
           'OUPS',
           'SMOTE_D',
           'SMOTE_PSO',
           'CURE_SMOTE',
           'SOMO',
           'ISOMAP_Hybrid',
           'CE_SMOTE',
           'Edge_Det_SMOTE',
           'CBSO',
           'E_SMOTE',
           'DBSMOTE',
           'ASMOBD',
           'Assembled_SMOTE',
           'SDSMOTE',
           'DSMOTE',
           'G_SMOTE',
           'NT_SMOTE',
           'Lee',
           'SPY',
           'SMOTE_PSOBAT',
           'MDO',
           'Random_SMOTE',
           'ISMOTE',
           'VIS_RST',
           'GASMOTE',
           'A_SUWO',
           'SMOTE_FRST_2T',
           'AND_SMOTE',
           'NRAS',
           'AMSCO',
           'SSO',
           'NDO_sampling',
           'DSRBF',
           'Gaussian_SMOTE',
           'kmeans_SMOTE',
           'Supervised_SMOTE',
           'SN_SMOTE',
           'CCR',
           'ANS',
           'cluster_SMOTE',
           'NoSMOTE',
           'MulticlassOversampling',
           'OversamplingClassifier']


def get_all_oversamplers():
    """
    Returns all oversampling classes

    Returns:
        list(OverSampling): list of all oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers()
    """

    return OverSampling.__subclasses__()


def get_n_quickest_oversamplers(n=10):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package.

    Args:
        n (int): number of oversamplers to return

    Returns:
        list(OverSampling): list of the n quickest oversampling classes

    Example::

        import smote_variants as sv

        oversamplers= sv.get_n_quickest_oversamplers(10)
    """

    runtimes = {'SPY': 0.11, 'OUPS': 0.16, 'SMOTE_D': 0.20, 'NT_SMOTE': 0.20,
                'Gazzah': 0.21, 'ROSE': 0.25, 'NDO_sampling': 0.27,
                'Borderline_SMOTE1': 0.28, 'SMOTE': 0.28,
                'Borderline_SMOTE2': 0.29, 'ISMOTE': 0.30, 'SMMO': 0.31,
                'SMOTE_OUT': 0.37, 'SN_SMOTE': 0.44, 'Selected_SMOTE': 0.47,
                'distance_SMOTE': 0.47, 'Gaussian_SMOTE': 0.48, 'MCT': 0.51,
                'Random_SMOTE': 0.57, 'ADASYN': 0.58, 'SL_graph_SMOTE': 0.58,
                'CURE_SMOTE': 0.59, 'ANS': 0.63, 'MSMOTE': 0.72,
                'Safe_Level_SMOTE': 0.79, 'SMOBD': 0.80, 'CBSO': 0.81,
                'Assembled_SMOTE': 0.82, 'SDSMOTE': 0.88,
                'SMOTE_TomekLinks': 0.91, 'Edge_Det_SMOTE': 0.94,
                'ProWSyn': 1.00, 'Stefanowski': 1.04, 'NRAS': 1.06,
                'AND_SMOTE': 1.13, 'DBSMOTE': 1.17, 'polynom_fit_SMOTE': 1.18,
                'ASMOBD': 1.18, 'MDO': 1.18, 'SOI_CJ': 1.24, 'LN_SMOTE': 1.26,
                'VIS_RST': 1.34, 'TRIM_SMOTE': 1.36, 'LLE_SMOTE': 1.62,
                'SMOTE_ENN': 1.86, 'SMOTE_Cosine': 2.00, 'kmeans_SMOTE': 2.43,
                'MWMOTE': 2.45, 'V_SYNTH': 2.59, 'A_SUWO': 2.81,
                'RWO_sampling': 2.91, 'SMOTE_RSB': 3.88, 'ADOMS': 3.89,
                'SMOTE_IPF': 4.10, 'Lee': 4.16, 'SMOTE_FRST_2T': 4.18,
                'cluster_SMOTE': 4.19, 'SOMO': 4.30, 'DE_oversampling': 4.67,
                'CCR': 4.72, 'NRSBoundary_SMOTE': 5.26, 'AHC': 5.27,
                'ISOMAP_Hybrid': 6.11, 'LVQ_SMOTE': 6.99, 'CE_SMOTE': 7.45,
                'MSYN': 11.92, 'PDFOS': 15.14, 'KernelADASYN': 17.87,
                'G_SMOTE': 19.23, 'E_SMOTE': 19.50, 'SVM_balance': 24.05,
                'SUNDO': 26.21, 'GASMOTE': 31.38, 'DEAGO': 33.39,
                'NEATER': 41.39, 'SMOTE_PSO': 45.12, 'IPADE_ID': 90.01,
                'DSMOTE': 146.73, 'MOT2LD': 149.42, 'Supervised_SMOTE': 195.74,
                'SSO': 215.27, 'DSRBF': 272.11, 'SMOTE_PSOBAT': 324.31,
                'ADG': 493.64, 'AMSCO': 1502.36}

    samplers = get_all_oversamplers()
    samplers = sorted(
        samplers, key=lambda x: runtimes.get(x.__name__, 1e8))

    return samplers[:n]


def get_all_oversamplers_multiclass(strategy="eq_1_vs_many_successive"):
    """
    Returns all oversampling classes which can be used with the multiclass
    strategy specified

    Args:
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'

    Returns:
        list(OverSampling): list of all oversampling classes which can be used
                            with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_all_oversamplers_multiclass()
    """

    oversamplers = get_all_oversamplers()

    if (strategy == 'eq_1_vs_many_successive' or
            strategy == 'equalize_1_vs_many'):

        def multiclass_filter(o):
            return ((OverSampling.cat_changes_majority not in o.categories) or
                    ('proportion' in o().get_params()))

        return [o for o in oversamplers if multiclass_filter(o)]
    else:
        raise ValueError(("It is not known which oversamplers work with the"
                          " strategy %s") % strategy)


def get_n_quickest_oversamplers_multiclass(n,
                                           strategy="eq_1_vs_many_successive"):
    """
    Returns the n quickest oversamplers based on testing on the datasets of
    the imbalanced_databases package, and suitable for using the multiclass
    strategy specified.

    Args:
        n (int): number of oversamplers to return
        strategy (str): the multiclass oversampling strategy -
                        'eq_1_vs_many_successive'/'equalize_1_vs_many'

    Returns:
        list(OverSampling): list of n quickest oversampling classes which can
                    be used with the multiclass strategy specified

    Example::

        import smote_variants as sv

        oversamplers= sv.get_n_quickest_oversamplers_multiclass()
    """

    oversamplers = get_all_oversamplers()
    quickest_oversamplers = get_n_quickest_oversamplers(len(oversamplers))

    if (strategy == 'eq_1_vs_many_successive'
            or strategy == 'equalize_1_vs_many'):

        def multiclass_filter(o):
            return ((OverSampling.cat_changes_majority not in o.categories) or
                    ('proportion' in o().get_params()))

        return [o for o in quickest_oversamplers if multiclass_filter(o)][:n]
    else:
        raise ValueError("It is not known which oversamplers work with the"
                         " strategy %s" % strategy)


def get_all_noisefilters():
    """
    Returns all noise filters
    Returns:
        list(NoiseFilter): list of all noise filter classes
    """
    return NoiseFilter.__subclasses__()


def mode(data):
    values, counts = np.unique(data, return_counts=True)
    return values[np.where(counts == max(counts))[0][0]]


class StatisticsMixin:
    """
    Mixin to compute class statistics and determine minority/majority labels
    """

    def class_label_statistics(self, X, y):
        """
        determines class sizes and minority and majority labels
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        unique, counts = np.unique(y, return_counts=True)
        self.class_stats = dict(zip(unique, counts))
        self.min_label = unique[0] if counts[0] < counts[1] else unique[1]
        self.maj_label = unique[1] if counts[0] < counts[1] else unique[0]
        # shorthands
        self.min_label = self.min_label
        self.maj_label = self.maj_label

    def check_enough_min_samples_for_sampling(self, threshold=2):
        if self.class_stats[self.min_label] < threshold:
            m = ("The number of minority samples (%d) is not enough "
                 "for sampling")
            m = m % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + m)
            return False
        return True


class RandomStateMixin:
    """
    Mixin to set random state
    """

    def set_random_state(self, random_state):
        """
        sets the random_state member of the object

        Args:
            random_state (int/np.random.RandomState/None): the random state
                                                                initializer
        """

        self._random_state_init = random_state

        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif random_state is np.random:
            self.random_state = random_state
        else:
            raise ValueError(
                "random state cannot be initialized by " + str(random_state))


class ParameterCheckingMixin:
    """
    Mixin to check if parameters come from a valid range
    """

    def check_in_range(self, x, name, r):
        """
        Check if parameter is in range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x < r[0] or x > r[1]:
            m = ("Value for parameter %s outside the range [%f,%f] not"
                 " allowed: %f")
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_out_range(self, x, name, r):
        """
        Check if parameter is outside of range
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            r (list-like(2)): the lower and upper bound of a range
        Throws:
            ValueError
        """
        if x >= r[0] and x <= r[1]:
            m = "Value for parameter %s in the range [%f,%f] not allowed: %f"
            m = m % (name, r[0], r[1], x)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal(self, x, name, val):
        """
        Check if parameter is less than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x > val:
            m = "Value for parameter %s greater than %f not allowed: %f > %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x > y:
            m = ("Value for parameter %s greater than parameter %s not"
                 " allowed: %f > %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less(self, x, name, val):
        """
        Check if parameter is less than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x >= val:
            m = ("Value for parameter %s greater than or equal to %f"
                 " not allowed: %f >= %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_less_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x >= y:
            m = ("Value for parameter %s greater than or equal to parameter"
                 " %s not allowed: %f >= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal(self, x, name, val):
        """
        Check if parameter is greater than or equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x < val:
            m = "Value for parameter %s less than %f is not allowed: %f < %f"
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_or_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is less than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x < y:
            m = ("Value for parameter %s less than parameter %s is not"
                 " allowed: %f < %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater(self, x, name, val):
        """
        Check if parameter is greater than value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x <= val:
            m = ("Value for parameter %s less than or equal to %f not allowed"
                 " %f < %f")
            m = m % (name, val, x, val)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_greater_par(self, x, name_x, y, name_y):
        """
        Check if parameter is greater than or equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x <= y:
            m = ("Value for parameter %s less than or equal to parameter %s"
                 " not allowed: %f <= %f")
            m = m % (name_x, name_y, x, y)

            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal(self, x, name, val):
        """
        Check if parameter is equal to value
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            val (numeric): value to compare to
        Throws:
            ValueError
        """
        if x == val:
            m = ("Value for parameter %s equal to parameter %f is not allowed:"
                 " %f == %f")
            m = m % (name, val, x, val)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_equal_par(self, x, name_x, y, name_y):
        """
        Check if parameter is equal to another parameter
        Args:
            x (numeric): the parameter value
            name_x (str): the parameter name
            y (numeric): the other parameter value
            name_y (str): the other parameter name
        Throws:
            ValueError
        """
        if x == y:
            m = ("Value for parameter %s equal to parameter %s is not "
                 "allowed: %f == %f")
            m = m % (name_x, name_y, x, y)
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_isin(self, x, name, li):
        """
        Check if parameter is in list
        Args:
            x (numeric): the parameter value
            name (str): the parameter name
            li (list): list to check if parameter is in it
        Throws:
            ValueError
        """
        if x not in li:
            m = "Value for parameter %s not in list %s is not allowed: %s"
            m = m % (name, str(li), str(x))
            raise ValueError(self.__class__.__name__ + ": " + m)

    def check_n_jobs(self, x, name):
        """
        Check n_jobs parameter
        Args:
            x (int/None): number of jobs
            name (str): the parameter name
        Throws:
            ValueError
        """
        if not ((x is None)
                or (x is not None and isinstance(x, int) and not x == 0)):
            m = "Value for parameter n_jobs is not allowed: %s" % str(x)
            raise ValueError(self.__class__.__name__ + ": " + m)


class ParameterCombinationsMixin:
    """
    Mixin to generate parameter combinations
    """

    @classmethod
    def generate_parameter_combinations(cls, dictionary, raw):
        """
        Generates reasonable paramter combinations
        Args:
            dictionary (dict): dictionary of paramter ranges
            num (int): maximum number of combinations to generate
        """
        if raw:
            return dictionary
        keys = sorted(list(dictionary.keys()))
        values = [dictionary[k] for k in keys]
        combinations = [dict(zip(keys, p))
                        for p in list(itertools.product(*values))]
        return combinations


class NoiseFilter(StatisticsMixin,
                  ParameterCheckingMixin,
                  ParameterCombinationsMixin):
    """
    Parent class of noise filtering methods
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        pass

    def get_params(self, deep=False):
        """
        Return parameters

        Returns:
            dict: dictionary of parameters
        """

        return {}

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self


class TomekLinkRemoval(NoiseFilter):
    """
    Tomek link removal

    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, strategy='remove_majority', n_jobs=1):
        """
        Constructor of the noise filter.

        Args:
            strategy (str): noise removal strategy:
                            'remove_majority'/'remove_both'
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_isin(strategy, 'strategy', [
                        'remove_majority', 'remove_both'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        # using 2 neighbors because the first neighbor is the point itself
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        distances, indices = nn.fit(X).kneighbors(X)

        # identify links
        links = []
        for i in range(len(indices)):
            if indices[indices[i][1]][1] == i:
                if not y[indices[i][1]] == y[indices[indices[i][1]][1]]:
                    links.append((i, indices[i][1]))

        # determine links to be removed
        to_remove = []
        for li in links:
            if self.strategy == 'remove_majority':
                if y[li[0]] == self.min_label:
                    to_remove.append(li[1])
                else:
                    to_remove.append(li[0])
            elif self.strategy == 'remove_both':
                to_remove.append(li[0])
                to_remove.append(li[1])
            else:
                m = 'No Tomek link strategy %s implemented' % self.strategy
                raise ValueError(self.__class__.__name__ + ": " + m)

        to_remove = list(set(to_remove))

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)


class CondensedNearestNeighbors(NoiseFilter):
    """
    Condensed nearest neighbors

    References:
        * BibTex::

            @ARTICLE{condensed_nn,
                        author={Hart, P.},
                        journal={IEEE Transactions on Information Theory},
                        title={The condensed nearest neighbor rule (Corresp.)},
                        year={1968},
                        volume={14},
                        number={3},
                        pages={515-516},
                        keywords={Pattern classification},
                        doi={10.1109/TIT.1968.1054155},
                        ISSN={0018-9448},
                        month={May}}
    """

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removing object

        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        # Initial result set consists of all minority samples and 1 majority
        # sample

        X_maj = X[y == self.maj_label]
        X_hat = np.vstack([X[y == self.min_label], X_maj[0]])
        y_hat = np.hstack([np.repeat(self.min_label, len(X_hat)-1),
                           [self.maj_label]])
        X_maj = X_maj[1:]

        # Adding misclassified majority elements repeatedly
        while True:
            knn = KNeighborsClassifier(n_neighbors=1, n_jobs=self.n_jobs)
            knn.fit(X_hat, y_hat)
            pred = knn.predict(X_maj)

            if np.all(pred == self.maj_label):
                break
            else:
                X_hat = np.vstack([X_hat, X_maj[pred != self.maj_label]])
                y_hat = np.hstack(
                    [y_hat,
                     np.repeat(self.maj_label, len(X_hat) - len(y_hat))])
                X_maj = np.delete(X_maj, np.where(
                    pred != self.maj_label)[0], axis=0)
                if len(X_maj) == 0:
                    break

        return X_hat, y_hat


class OneSidedSelection(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods
                                for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        t = TomekLinkRemoval(n_jobs=self.n_jobs)
        X0, y0 = t.remove_noise(X, y)
        cnn = CondensedNearestNeighbors(n_jobs=self.n_jobs)

        return cnn.remove_noise(X0, y0)


class CNNTomekLinks(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods
                                for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        c = CondensedNearestNeighbors(n_jobs=self.n_jobs)
        X0, y0 = c.remove_noise(X, y)
        t = TomekLinkRemoval(n_jobs=self.n_jobs)

        return t.remove_noise(X0, y0)


class NeighborhoodCleaningRule(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        # fitting nearest neighbors with proposed parameter
        # using 4 neighbors because the first neighbor is the point itself
        nn = NearestNeighbors(n_neighbors=4, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # identifying the samples to be removed
        to_remove = []
        for i in range(len(X)):
            if (y[i] == self.maj_label and
                    mode(y[indices[i][1:]]) == self.min_label):
                # if sample i is majority and the decision based on
                # neighbors is minority
                to_remove.append(i)
            elif (y[i] == self.min_label and
                  mode(y[indices[i][1:]]) == self.maj_label):
                # if sample i is minority and the decision based on
                # neighbors is majority
                for j in indices[i][1:]:
                    if y[j] == self.maj_label:
                        to_remove.append(j)

        # removing the noisy samples and returning the results
        to_remove = list(set(to_remove))
        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)


class EditedNearestNeighbors(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, remove='both', n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            remove (str): class to remove from 'both'/'min'/'maj'
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_isin(remove, 'remove', ['both', 'min', 'maj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.remove = remove
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        if len(X) < 4:
            _logger.info(self.__class__.__name__ + ': ' +
                         "Not enough samples for noise removal")
            return X.copy(), y.copy()

        nn = NearestNeighbors(n_neighbors=4, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        to_remove = []
        for i in range(len(X)):
            if not y[i] == mode(y[indices[i][1:]]):
                if (self.remove == 'both' or
                    (self.remove == 'min' and y[i] == self.min_label) or
                        (self.remove == 'maj' and y[i] == self.maj_label)):
                    to_remove.append(i)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self):
        """
        Get noise removal parameters

        Returns:
            dict: dictionary of parameters
        """
        return {'remove': self.remove}


class OverSampling(StatisticsMixin,
                   ParameterCheckingMixin,
                   ParameterCombinationsMixin,
                   RandomStateMixin):
    """
    Base class of oversampling methods
    """

    categories = []

    cat_noise_removal = 'NR'
    cat_dim_reduction = 'DR'
    cat_uses_classifier = 'Clas'
    cat_sample_componentwise = 'SCmp'
    cat_sample_ordinary = 'SO'
    cat_sample_copy = 'SCpy'
    cat_memetic = 'M'
    cat_density_estimation = 'DE'
    cat_density_based = 'DB'
    cat_extensive = 'Ex'
    cat_changes_majority = 'CM'
    cat_uses_clustering = 'Clus'
    cat_borderline = 'BL'
    cat_application = 'A'

    def __init__(self):
        pass

    def det_n_to_sample(self, strategy, n_maj, n_min):
        """
        Determines the number of samples to generate
        Args:
            strategy (str/float): if float, the fraction of the difference
                                    of the minority and majority numbers to
                                    generate, like 0.1 means that 10% of the
                                    difference will be generated if str,
                                    like 'min2maj', the minority class will
                                    be upsampled to match the cardinality
                                    of the majority class
        """
        if isinstance(strategy, float) or isinstance(strategy, int):
            return max([0, int((n_maj - n_min)*strategy)])
        else:
            m = "Value %s for parameter strategy is not supported" % strategy
            raise ValueError(self.__class__.__name__ + ": " + m)

    def sample_between_points(self, x, y):
        """
        Sample randomly along the line between two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
        Returns:
            np.array: the new sample
        """
        return x + (y - x)*self.random_state.random_sample()

    def sample_between_points_componentwise(self, x, y, mask=None):
        """
        Sample each dimension separately between the two points.
        Args:
            x (np.array): point 1
            y (np.array): point 2
            mask (np.array): array of 0,1s - specifies which dimensions
                                to sample
        Returns:
            np.array: the new sample being generated
        """
        if mask is None:
            return x + (y - x)*self.random_state.random_sample()
        else:
            return x + (y - x)*self.random_state.random_sample()*mask

    def sample_by_jittering(self, x, std):
        """
        Sample by jittering.
        Args:
            x (np.array): base point
            std (float): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample() - 0.5)*2.0*std

    def sample_by_jittering_componentwise(self, x, std):
        """
        Sample by jittering componentwise.
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return x + (self.random_state.random_sample(len(x))-0.5)*2.0 * std

    def sample_by_gaussian_jittering(self, x, std):
        """
        Sample by Gaussian jittering
        Args:
            x (np.array): base point
            std (np.array): standard deviation
        Returns:
            np.array: the new sample
        """
        return self.random_state.normal(x, std)

    def sample(self, X, y):
        """
        The samplig function reimplemented in child classes
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        return X, y

    def fit_resample(self, X, y):
        """
        Alias of the function "sample" for compatibility with imbalanced-learn
        pipelines
        """
        return self.sample(X, y)

    def sample_with_timing(self, X, y):
        begin = time.time()
        X_samp, y_samp = self.sample(X, y)
        _logger.info(self.__class__.__name__ + ": " +
                     ("runtime: %f" % (time.time() - begin)))
        return X_samp, y_samp

    def preprocessing_transform(self, X):
        """
        Transforms new data according to the possible transformation
        implemented by the function "sample".
        Args:
            X (np.matrix): features
        Returns:
            np.matrix: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))

    def __str__(self):
        return self.descriptor()


class UnderSampling(StatisticsMixin,
                    ParameterCheckingMixin,
                    ParameterCombinationsMixin):
    """
    Base class of undersampling approaches.
    """

    def __init__(self):
        """
        Constructorm
        """
        super().__init__()

    def sample(self, X, y):
        """
        Carry out undersampling
        Args:
            X (np.matrix): features
            y (np.array): labels
        Returns:
            np.matrix, np.array: sampled X and y
        """
        pass

    def get_params(self, deep=False):
        """
        Returns the parameters of the object as a dictionary.
        Returns:
            dict: the parameters of the object
        """
        pass

    def descriptor(self):
        """
        Returns:
            str: JSON description of the current sampling object
        """
        return str((self.__class__.__name__, str(self.get_params())))


class NoSMOTE(OverSampling):
    """
    The goal of this class is to provide a functionality to send data through
    on any model selection/evaluation pipeline with no oversampling carried
    out. It can be used to get baseline estimates on preformance.
    """

    categories = []

    def __init__(self, random_state=None):
        """
        Constructor of the NoSMOTE object.

        random_state (int/np.random.RandomState/None): dummy parameter for the
                        compatibility of interfaces
        """
        super().__init__()

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({}, raw=False)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {}


class SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{smote,
                author={Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and
                            Kegelmeyer, W. P.},
                title={{SMOTE}: synthetic minority over-sampling technique},
                journal={Journal of Artificial Intelligence Research},
                volume={16},
                year={2002},
                pages={321--357}
              }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            # _logger.warning(self.__class__.__name__ +
            #                ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the model
        n_neigh = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if n_to_sample == 0:
            return X.copy(), y.copy()

        # generating samples
        base_indices = self.random_state.choice(list(range(len(X_min))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, n_neigh)),
                                                    n_to_sample)

        X_base = X_min[base_indices]
        X_neighbor = X_min[ind[base_indices, neighbor_indices]]

        samples = X_base + np.multiply(self.random_state.rand(n_to_sample,
                                                              1),
                                       X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_TomekLinks(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_tomeklinks_enn,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA},
                    }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return SMOTE.parameter_combinations(raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        smote = SMOTE(self.proportion,
                      self.n_neighbors,
                      n_jobs=self.n_jobs,
                      random_state=self.random_state)
        X_new, y_new = smote.sample(X, y)

        t = TomekLinkRemoval(strategy='remove_both', n_jobs=self.n_jobs)

        X_samp, y_samp = t.remove_noise(X_new, y_new)

        if len(X_samp) == 0:
            m = ("All samples have been removed, "
                 "returning the original dataset.")
            _logger.info(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        return X_samp, y_samp

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_ENN(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_tomeklinks_enn,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA},
                    }

    Notes:
        * Can remove too many of minority samples.
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return SMOTE.parameter_combinations(raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        smote = SMOTE(self.proportion, self.n_neighbors,
                      n_jobs=self.n_jobs, random_state=self.random_state)
        X_new, y_new = smote.sample(X, y)

        enn = EditedNearestNeighbors(n_jobs=self.n_jobs)

        return enn.remove_noise(X_new, y_new)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Borderline_SMOTE1(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling Method
                                     in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): control parameter of the nearest neighbor
                                    technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                    technique for sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # fitting model
        X_min = X[y == self.min_label]

        n_neighbors = min([len(X), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_min)

        # determining minority samples in danger
        noise = []
        danger = []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.maj_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.maj_label:
                danger.append(i)
        X_danger = X_min[danger]
        X_min = np.delete(X_min, np.array(noise).astype(int), axis=0)

        if len(X_danger) == 0:
            _logger.info(self.__class__.__name__ +
                         ": " + "No samples in danger")
            return X.copy(), y.copy()

        # fitting nearest neighbors model to minority samples
        k_neigh = min([len(X_min), self.k_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=k_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        # extracting neighbors of samples in danger
        distances, indices = nn.kneighbors(X_danger)

        # generating samples near points in danger
        base_indices = self.random_state.choice(list(range(len(X_danger))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, k_neigh)),
                                                    n_to_sample)

        X_base = X_danger[base_indices]
        X_neighbor = X_min[indices[base_indices, neighbor_indices]]

        samples = X_base + \
            np.multiply(self.random_state.rand(
                n_to_sample, 1), X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Borderline_SMOTE2(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling
                                    Method in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                technique for sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # fitting nearest neighbors model
        X_min = X[y == self.min_label]

        n_neighbors = min([self.n_neighbors+1, len(X)])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_min)

        # determining minority samples in danger
        noise = []
        danger = []
        for i in range(len(indices)):
            if self.n_neighbors == sum(y[indices[i][1:]] == self.maj_label):
                noise.append(i)
            elif mode(y[indices[i][1:]]) == self.maj_label:
                danger.append(i)
        X_danger = X_min[danger]
        X_min = np.delete(X_min, np.array(noise).astype(int), axis=0)

        if len(X_min) < 2:
            m = ("The number of minority samples after preprocessing (%d) is "
                 "not enough for sampling")
            m = m % (len(X_min))
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        if len(X_danger) == 0:
            m = "No samples in danger"
            _logger.info(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # fitting nearest neighbors model to minority samples
        k_neigh = self.k_neighbors + 1
        k_neigh = min([k_neigh, len(X)])
        nn = NearestNeighbors(n_neighbors=k_neigh, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_danger)

        # generating the samples
        base_indices = self.random_state.choice(
            list(range(len(X_danger))), n_to_sample)
        neighbor_indices = self.random_state.choice(
            list(range(1, k_neigh)), n_to_sample)

        X_base = X_danger[base_indices]
        X_neighbor = X[indices[base_indices, neighbor_indices]]
        diff = X_neighbor - X_base
        r = self.random_state.rand(n_to_sample, 1)
        mask = y[neighbor_indices] == self.maj_label
        r[mask] = r[mask]*0.5

        samples = X_base + np.multiply(r, diff)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ADASYN(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{adasyn,
                          author={He, H. and Bai, Y. and Garcia,
                                    E. A. and Li, S.},
                          title={{ADASYN}: adaptive synthetic sampling
                                    approach for imbalanced learning},
                          booktitle={Proceedings of IJCNN},
                          year={2008},
                          pages={1322--1328}
                        }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_density_based]

    def __init__(self,
                 n_neighbors=5,
                 d_th=0.9,
                 beta=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            d_th (float): tolerated deviation level from balancedness
            beta (float): target level of balancedness, same as proportion
                            in other techniques
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(d_th, 'd_th', 0)
        self.check_greater_or_equal(beta, 'beta', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.d_th = d_th
        self.beta = beta
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7, 9],
                                  'd_th': [0.9],
                                  'beta': [1.0, 0.75, 0.5, 0.25]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # extracting minority samples
        X_min = X[y == self.min_label]

        # checking if sampling is needed
        m_min = len(X_min)
        m_maj = len(X) - m_min

        n_to_sample = (m_maj - m_min)*self.beta

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        d = float(m_min)/m_maj
        if d > self.d_th:
            return X.copy(), y.copy()

        # fitting nearest neighbors model to all samples
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_min)

        # determining the distribution of points to be generated
        r = []
        for i in range(len(indices)):
            r.append(sum(y[indices[i][1:]] ==
                         self.maj_label)/self.n_neighbors)
        r = np.array(r)
        if sum(r) > 0:
            r = r/sum(r)

        if any(np.isnan(r)) or sum(r) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "not enough non-noise samples for oversampling")
            return X.copy(), y.copy()

        # fitting nearest neighbors models to minority samples
        n_neigh = min([len(X_min), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=self.n_jobs)
        nn.fit(X_min)
        distances, indices = nn.kneighbors(X_min)

        # sampling points
        base_indices = self.random_state.choice(
            list(range(len(X_min))), size=int(n_to_sample), p=r)
        neighbor_indices = self.random_state.choice(
            list(range(1, n_neigh)), int(n_to_sample))

        X_base = X_min[base_indices]
        X_neighbor = X_min[indices[base_indices, neighbor_indices]]
        diff = X_neighbor - X_base
        r = self.random_state.rand(int(n_to_sample), 1)

        samples = X_base + np.multiply(r, diff)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*int(n_to_sample))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'd_th': self.d_th,
                'beta': self.beta,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class AHC(OverSampling):
    """
    References:
        * BibTex::

            @article{AHC,
                    title = "Learning from imbalanced data in surveillance
                             of nosocomial infection",
                    journal = "Artificial Intelligence in Medicine",
                    volume = "37",
                    number = "1",
                    pages = "7 - 18",
                    year = "2006",
                    note = "Intelligent Data Analysis in Medicine",
                    issn = "0933-3657",
                    doi = "https://doi.org/10.1016/j.artmed.2005.03.002",
                    url = {http://www.sciencedirect.com/science/article/
                            pii/S0933365705000850},
                    author = "Gilles Cohen and Mélanie Hilario and Hugo Sax
                                and Stéphane Hugonnet and Antoine Geissbuhler",
                    keywords = "Nosocomial infection, Machine learning,
                                    Support vector machines, Data imbalance"
                    }
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_application]

    def __init__(self, strategy='min', n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            strategy (str): which class to sample (min/maj/minmaj)
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_isin(strategy, 'strategy', ['min', 'maj', 'minmaj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'strategy': ['min', 'maj', 'minmaj']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample_majority(self, X, n_clusters):
        """
        Sample the majority class

        Args:
            X (np.ndarray): majority samples
            n_clusters (int): number of clusters to find

        Returns:
            np.ndarray: downsampled vectors
        """
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X)
        return kmeans.cluster_centers_

    def sample_minority(self, X):
        """
        Sampling the minority class

        Args:
            X (np.ndarray): minority samples

        Returns:
            np.ndarray: the oversampled set of vectors
        """
        ac = AgglomerativeClustering(n_clusters=1)
        ac.fit(X)
        n_samples = len(X)

        cc = [None]*len(ac.children_)
        weights = [None]*len(ac.children_)

        def cluster_centers(children, i, cc, weights):
            """
            Extract cluster centers

            Args:
                children (np.array): indices of children
                i (int): index to process
                cc (np.array): cluster centers
                weights (np.array): cluster weights

            Returns:
                int, float: new cluster center, new weight
            """
            if i < n_samples:
                return X[i], 1.0

            if cc[i - n_samples] is None:
                a, w_a = cluster_centers(
                    children, children[i - n_samples][0], cc, weights)
                b, w_b = cluster_centers(
                    children, children[i - n_samples][1], cc, weights)
                cc[i - n_samples] = (w_a*a + w_b*b)/(w_a + w_b)
                weights[i - n_samples] = w_a + w_b

            return cc[i - n_samples], weights[i - n_samples]

        cluster_centers(ac.children_, ac.children_[-1][-1] + 1, cc, weights)

        return np.vstack(cc)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # extracting minority samples
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        if self.strategy == 'maj':
            X_maj_resampled = self.sample_majority(X_maj, len(X_min))
            return (np.vstack([X_min, X_maj_resampled]),
                    np.hstack([np.repeat(self.min_label, len(X_min)),
                               np.repeat(self.maj_label,
                                         len(X_maj_resampled))]))
        elif self.strategy == 'min':
            X_min_resampled = self.sample_minority(X_min)
            return (np.vstack([X_min_resampled, X_min, X_maj]),
                    np.hstack([np.repeat(self.min_label,
                                         (len(X_min_resampled) + len(X_min))),
                               np.repeat(self.maj_label, len(X_maj))]))
        elif self.strategy == 'minmaj':
            X_min_resampled = self.sample_minority(X_min)
            n_maj_sample = min([len(X_maj), len(X_min_resampled) + len(X_min)])
            X_maj_resampled = self.sample_majority(X_maj, n_maj_sample)
            return (np.vstack([X_min_resampled, X_min, X_maj_resampled]),
                    np.hstack([np.repeat(self.min_label,
                                         (len(X_min_resampled) + len(X_min))),
                               np.repeat(self.maj_label,
                                         len(X_maj_resampled))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'strategy': self.strategy,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class LLE_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{lle_smote,
                            author={Wang, J. and Xu, M. and Wang,
                                    H. and Zhang, J.},
                            booktitle={2006 8th international Conference
                                    on Signal Processing},
                            title={Classification of Imbalanced Data by Using
                                    the SMOTE Algorithm and Locally Linear
                                    Embedding},
                            year={2006},
                            volume={3},
                            number={},
                            pages={},
                            keywords={artificial intelligence;
                                        biomedical imaging;medical computing;
                                        imbalanced data classification;
                                        SMOTE algorithm;
                                        locally linear embedding;
                                        medical imaging intelligence;
                                        synthetic minority oversampling
                                        technique;
                                        high-dimensional data;
                                        low-dimensional space;
                                        Biomedical imaging;
                                        Back;Training data;
                                        Data mining;Biomedical engineering;
                                        Research and development;
                                        Electronic mail;Pattern recognition;
                                        Performance analysis;
                                        Classification algorithms},
                            doi={10.1109/ICOSP.2006.345752},
                            ISSN={2164-5221},
                            month={Nov}}

    Notes:
        * There might be numerical issues if the nearest neighbors contain
            some element multiple times.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_components=2,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj
                                and n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples will
                                be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            n_components (int): dimensionality of the embedding space
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 2)
        self.check_greater_or_equal(n_components, 'n_components', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_components': [2, 3, 5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # extracting minority samples
        X_min = X[y == self.min_label]

        # do the locally linear embedding
        lle = LocallyLinearEmbedding(
            self.n_neighbors, self.n_components, n_jobs=self.n_jobs)
        try:
            lle.fit(X_min)
        except Exception as e:
            return X.copy(), y.copy()
        X_min_transformed = lle.transform(X_min)

        # fitting the nearest neighbors model for sampling
        n_neighbors = min([self.n_neighbors+1, len(X_min_transformed)])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs).fit(X_min_transformed)
        dist, ind = nn.kneighbors(X_min_transformed)

        def solve_for_weights(xi, Z):
            """
            Solve for locally linear embedding weights

            Args:
                xi (np.array): vector
                Z (np.matrix): matrix of neighbors in rows

            Returns:
                np.array: reconstruction weights

            Following https://cs.nyu.edu/~roweis/lle/algorithm.html
            """
            Z = Z - xi
            Z = Z.T
            C = np.dot(Z.T, Z)
            try:
                w = np.linalg.solve(C, np.repeat(1.0, len(C)))
                if np.linalg.norm(w) > 1e8:
                    w = np.repeat(1.0, len(C))
            except Exception as e:
                w = np.repeat(1.0, len(C))
            return w/np.sum(w)

        # generating samples
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X_min))
            random_coords = self.random_state.choice(ind[idx][1:])
            xi = self.sample_between_points(X_min_transformed[idx],
                                            X_min_transformed[random_coords])
            Z = X_min_transformed[ind[idx][1:]]
            w = solve_for_weights(xi, Z)
            samples.append(np.dot(w, X_min[ind[idx][1:]]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_components': self.n_components,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class distance_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{distance_smote,
                            author={de la Calleja, J. and Fuentes, O.},
                            booktitle={Proceedings of the Twentieth
                                        International Florida Artificial
                                        Intelligence},
                            title={A distance-based over-sampling method
                                    for learning from imbalanced data sets},
                            year={2007},
                            volume={3},
                            pages={634--635}
                            }

    Notes:
        * It is not clear what the authors mean by "weighted distance".
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # extracting minority samples
        X_min = X[y == self.min_label]

        # fitting the model
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X_min))
            mean_vector = np.mean(X_min[ind[idx][1:]], axis=0)
            samples.append(self.sample_between_points(X_min[idx], mean_vector))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMMO(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{smmo,
                            author = {de la Calleja, Jorge and Fuentes, Olac
                                        and González, Jesús},
                            booktitle= {Proceedings of the Twenty-First
                                        International Florida Artificial
                                        Intelligence Research Society
                                        Conference},
                            year = {2008},
                            month = {01},
                            pages = {276-281},
                            title = {Selecting Minority Examples from
                                    Misclassified Data for Over-Sampling.}
                            }

    Notes:
        * In this paper the ensemble is not specified. I have selected
            some very fast, basic classifiers.
        * Also, it is not clear what the authors mean by "weighted distance".
        * The original technique is not prepared for the case when no minority
            samples are classified correctly be the ensemble.
    """

    categories = [OverSampling.cat_borderline,
                  OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 ensemble=[QuadraticDiscriminantAnalysis(),
                           DecisionTreeClassifier(random_state=2),
                           GaussianNB()],
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            ensemble (list): list of classifiers, if None, default list of
                                classifiers is used
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        try:
            len_ens = len(ensemble)
        except Exception as e:
            raise ValueError('The ensemble needs to be a list-like object')
        if len_ens == 0:
            raise ValueError('At least 1 classifier needs to be specified')
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.ensemble = ensemble
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        ensembles = [[QuadraticDiscriminantAnalysis(),
                      DecisionTreeClassifier(random_state=2),
                      GaussianNB()]]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'ensemble': ensembles}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # training and in-sample prediction (out-of-sample by k-fold cross
        # validation might be better)
        predictions = []
        for e in self.ensemble:
            predictions.append(e.fit(X, y).predict(X))

        # constructing ensemble prediction
        pred = np.where(np.sum(np.vstack(predictions), axis=0)
                        > len(self.ensemble)/2, 1, 0)

        # create mask of minority samples to sample
        mask_to_sample = np.where(np.logical_and(np.logical_not(
            np.equal(pred, y)), y == self.min_label))[0]
        if len(mask_to_sample) < 2:
            m = "Not enough minority samples selected %d" % len(mask_to_sample)
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_min_to_sample = X[mask_to_sample]

        # fitting nearest neighbors model for sampling
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min_to_sample)

        # doing the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min_to_sample))
            mean = np.mean(X_min[ind[idx][1:]], axis=0)
            samples.append(self.sample_between_points(
                X_min_to_sample[idx], mean))

        return (np.vstack([X, np.vstack([samples])]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'ensemble': self.ensemble,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class polynom_fit_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            keywords={curve fitting;learning (artificial
                                        intelligence);mesh generation;pattern
                                        classification;polynomials;sampling
                                        methods;support vector machines;
                                        oversampling approach;polynomial
                                        fitting function;imbalanced data
                                        set;pattern classification task;
                                        class-modular strategy;support
                                        vector machine;true negative rate;
                                        true positive rate;star topology;
                                        bus topology;polynomial curve
                                        topology;mesh topology;Polynomials;
                                        Topology;Support vector machines;
                                        Support vector machine classification;
                                        Pattern classification;Performance
                                        evaluation;Training data;Text
                                        analysis;Data engineering;Convergence;
                                        writer identification system;majority
                                        class;minority class;imbalanced data
                                        sets;polynomial fitting functions;
                                        class-modular strategy},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 topology='star',
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            topoplogy (str): 'star'/'bus'/'mesh'
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        if topology.startswith('poly'):
            self.check_greater_or_equal(
                int(topology.split('_')[-1]), 'topology', 1)
        else:
            self.check_isin(topology, "topology", ['star', 'bus', 'mesh'])

        self.proportion = proportion
        self.topology = topology

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'topology': ['star', 'bus', 'mesh',
                                               'poly_1', 'poly_2', 'poly_3']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # extracting minority samples
        X_min = X[y == self.min_label]

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        samples = []
        if self.topology == 'star':
            # Implementation of the star topology
            X_mean = np.mean(X_min, axis=0)
            k = max([1, int(np.rint(n_to_sample/len(X_min)))])
            for x in X_min:
                diff = X_mean - x
                for i in range(1, k+1):
                    samples.append(x + float(i)/(k+1)*diff)
        elif self.topology == 'bus':
            # Implementation of the bus topology
            k = max([1, int(np.rint(n_to_sample/len(X_min)))])
            for i in range(1, len(X_min)):
                diff = X_min[i-1] - X_min[i]
                for j in range(1, k+1):
                    samples.append(X_min[i] + float(j)/(k+1)*diff)
        elif self.topology == 'mesh':
            # Implementation of the mesh topology
            if len(X_min)**2 > n_to_sample:
                while len(samples) < n_to_sample:
                    random_i = self.random_state.randint(len(X_min))
                    random_j = self.random_state.randint(len(X_min))
                    diff = X_min[random_i] - X_min[random_j]
                    samples.append(X_min[random_i] + 0.5*diff)
            else:
                n_combs = (len(X_min)*(len(X_min)-1)/2)
                k = max([1, int(np.rint(n_to_sample/n_combs))])
                for i in range(len(X_min)):
                    for j in range(len(X_min)):
                        diff = X_min[i] - X_min[j]
                        for li in range(1, k+1):
                            samples.append(X_min[j] + float(li)/(k+1)*diff)
        elif self.topology.startswith('poly'):
            # Implementation of the polynomial topology
            deg = int(self.topology.split('_')[1])
            dim = len(X_min[0])

            def fit_poly(d):
                return np.poly1d(np.polyfit(np.arange(len(X_min)),
                                            X_min[:, d], deg))

            polys = [fit_poly(d) for d in range(dim)]

            for d in range(dim):
                random_sample = self.random_state.random_sample()*len(X_min)
                samples_gen = [polys[d](random_sample)
                               for _ in range(n_to_sample)]
                samples.append(np.array(samples_gen))
            samples = np.vstack(samples).T

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'topology': self.topology,
                'random_state': self._random_state_init}


class Stefanowski(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{stefanowski,
                 author = {Stefanowski, Jerzy and Wilk, Szymon},
                 title = {Selective Pre-processing of Imbalanced Data for
                            Improving Classification Performance},
                 booktitle = {Proceedings of the 10th International Conference
                                on Data Warehousing and Knowledge Discovery},
                 series = {DaWaK '08},
                 year = {2008},
                 isbn = {978-3-540-85835-5},
                 location = {Turin, Italy},
                 pages = {283--292},
                 numpages = {10},
                 url = {http://dx.doi.org/10.1007/978-3-540-85836-2_27},
                 doi = {10.1007/978-3-540-85836-2_27},
                 acmid = {1430591},
                 publisher = {Springer-Verlag},
                 address = {Berlin, Heidelberg},
                }
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_sample_copy,
                  OverSampling.cat_borderline]

    def __init__(self, strategy='weak_amp', n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            strategy (str): 'weak_amp'/'weak_amp_relabel'/'strong_amp'
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_isin(strategy,
                        'strategy',
                        ['weak_amp', 'weak_amp_relabel', 'strong_amp'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.n_jobs = n_jobs

        # this method does not maintain randomness, the parameter is
        # introduced for the compatibility of interfaces
        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        if not raw:
            return [{'strategy': 'weak_amp'},
                    {'strategy': 'weak_amp_relabel'},
                    {'strategy': 'strong_amp'}, ]
        else:
            return {'strategy': ['weak_amp', 'weak_amp_relabel', 'strong_amp']}

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if self.class_stats[self.min_label] < 6:
            m = ("The number of minority samples (%d) is not"
                 " enough for sampling")
            m = m % (self.class_stats[self.min_label])
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # copying y as its values will change
        y = y.copy()
        # fitting the nearest neighbors model for noise filtering, 4 neighbors
        # instead of 3 as the closest neighbor to a point is itself
        nn = NearestNeighbors(n_neighbors=min(4, len(X)), n_jobs=self.n_jobs)
        nn.fit(X)
        distance, indices = nn.kneighbors(X)

        # fitting the nearest neighbors model for sample generation,
        # 6 neighbors instead of 5 for the same reason
        nn5 = NearestNeighbors(n_neighbors=min(6, len(X)), n_jobs=self.n_jobs)
        nn5.fit(X)
        distance5, indices5 = nn5.kneighbors(X)

        # determining noisy and safe flags
        flags = []
        for i in range(len(indices)):
            if mode(y[indices[i][1:]]) == y[i]:
                flags.append('safe')
            else:
                flags.append('noisy')
        flags = np.array(flags)

        D = (y == self.maj_label) & (flags == 'noisy')
        minority_indices = np.where(y == self.min_label)[0]

        samples = []
        if self.strategy == 'weak_amp' or self.strategy == 'weak_amp_relabel':
            # weak mplification - the number of copies is the number of
            # majority nearest neighbors
            for i in minority_indices:
                if flags[i] == 'noisy':
                    k = np.sum(np.logical_and(
                        y[indices[i][1:]] == self.maj_label,
                        flags[indices[i][1:]] == 'safe'))
                    for _ in range(k):
                        samples.append(X[i])
        if self.strategy == 'weak_amp_relabel':
            # relabling - noisy majority neighbors are relabelled to minority
            for i in minority_indices:
                if flags[i] == 'noisy':
                    for j in indices[i][1:]:
                        if y[j] == self.maj_label and flags[j] == 'noisy':
                            y[j] = self.min_label
                            D[j] = False
        if self.strategy == 'strong_amp':
            # safe minority samples are copied as many times as many safe
            # majority samples are among the nearest neighbors
            for i in minority_indices:
                if flags[i] == 'safe':
                    k = np.sum(np.logical_and(
                        y[indices[i][1:]] == self.maj_label,
                        flags[indices[i][1:]] == 'safe'))
                    for _ in range(k):
                        samples.append(X[i])
            # if classified correctly by knn(5), noisy minority samples are
            # amplified by creating as many copies as many save majority
            # samples in its neighborhood are present otherwise amplify
            # based on the 5 neighborhood
            for i in minority_indices:
                if flags[i] == 'noisy':
                    if mode(y[indices5[i][1:]]) == y[i]:
                        k = np.sum(np.logical_and(
                            y[indices[i][1:]] == self.maj_label,
                            flags[indices[i][1:]] == 'safe'))
                    else:
                        k = np.sum(np.logical_and(
                            y[indices5[i][1:]] == self.maj_label,
                            flags[indices5[i][1:]] == 'safe'))
                    for _ in range(k):
                        samples.append(X[i])

        to_remove = np.where(D)[0]

        X_noise_removed = np.delete(X, to_remove, axis=0)
        y_noise_removed = np.delete(y, to_remove, axis=0)

        if len(samples) == 0 and len(X_noise_removed) > 10:
            m = "no samples to add"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X_noise_removed, y_noise_removed
        elif len(samples) == 0:
            m = "all samples removed as noise, returning the original dataset"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        return (np.vstack([X_noise_removed,
                           np.vstack(samples)]),
                np.hstack([y_noise_removed,
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'strategy': self.strategy,
                'n_jobs': self.n_jobs}


class ADOMS(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{adoms,
                            author={Tang, S. and Chen, S.},
                            booktitle={2008 International Conference on
                                        Information Technology and
                                        Applications in Biomedicine},
                            title={The generation mechanism of synthetic
                                    minority class examples},
                            year={2008},
                            volume={},
                            number={},
                            pages={444-447},
                            keywords={medical image processing;
                                        generation mechanism;synthetic
                                        minority class examples;class
                                        imbalance problem;medical image
                                        analysis;oversampling algorithm;
                                        Principal component analysis;
                                        Biomedical imaging;Medical
                                        diagnostic imaging;Information
                                        technology;Biomedical engineering;
                                        Noise generators;Concrete;Nearest
                                        neighbor searches;Data analysis;
                                        Image analysis},
                            doi={10.1109/ITAB.2008.4570642},
                            ISSN={2168-2194},
                            month={May}}
    """

    categories = [OverSampling.cat_dim_reduction,
                  OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): parameter of the nearest neighbor component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0.0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        distance, indices = nn.kneighbors(X_min)

        samples = []
        for _ in range(n_to_sample):
            index = self.random_state.randint(len(X_min))
            neighbors = X_min[indices[index]]

            # fitting the PCA
            pca = PCA(n_components=1)
            pca.fit(neighbors)

            # extracting the principal direction
            principal_direction = pca.components_[0]

            # do the sampling according to the description in the paper
            random_index = self.random_state.randint(1, len(neighbors))
            random_neighbor = neighbors[random_index]
            d = np.linalg.norm(random_neighbor - X_min[index])
            r = self.random_state.random_sample()
            inner_product = np.dot(random_neighbor - X_min[index],
                                   principal_direction)
            sign = 1.0 if inner_product > 0.0 else -1.0
            samples.append(X_min[index] + sign*r*d*principal_direction)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Safe_Level_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{safe_level_smote,
                        author = {
                            Bunkhumpornpat, Chumphol and Sinapiromsaran,
                        Krung and Lursinsap, Chidchanok},
                        title = {Safe-Level-SMOTE: Safe-Level-Synthetic
                                Minority Over-Sampling TEchnique for
                                Handling the Class Imbalanced Problem},
                        booktitle = {Proceedings of the 13th Pacific-Asia
                                    Conference on Advances in Knowledge
                                    Discovery and Data Mining},
                        series = {PAKDD '09},
                        year = {2009},
                        isbn = {978-3-642-01306-5},
                        location = {Bangkok, Thailand},
                        pages = {475--482},
                        numpages = {8},
                        url = {http://dx.doi.org/10.1007/978-3-642-01307-2_43},
                        doi = {10.1007/978-3-642-01307-2_43},
                        acmid = {1533904},
                        publisher = {Springer-Verlag},
                        address = {Berlin, Heidelberg},
                        keywords = {Class Imbalanced Problem, Over-sampling,
                                    SMOTE, Safe Level},
                    }

    Notes:
        * The original method was not prepared for the case when no minority
            sample has minority neighbors.
    """

    categories = [OverSampling.cat_borderline,
                  OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1.0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # fitting nearest neighbors model
        n_neighbors = min([self.n_neighbors+1, len(X)])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distance, indices = nn.kneighbors(X)

        minority_labels = (y == self.min_label)
        minority_indices = np.where(minority_labels)[0]

        # do the sampling
        numattrs = len(X[0])
        samples = []
        for _ in range(n_to_sample):
            index = self.random_state.randint(len(minority_indices))
            neighbor_index = self.random_state.choice(indices[index][1:])

            p = X[index]
            n = X[neighbor_index]

            # find safe levels
            sl_p = np.sum(y[indices[index][1:]] == self.min_label)
            sl_n = np.sum(y[indices[neighbor_index][1:]]
                          == self.min_label)

            if sl_n > 0:
                sl_ratio = float(sl_p)/sl_n
            else:
                sl_ratio = np.inf

            if sl_ratio == np.inf and sl_p == 0:
                pass
            else:
                s = np.zeros(numattrs)
                for atti in range(numattrs):
                    # iterate through attributes and do sampling according to
                    # safe level
                    if sl_ratio == np.inf and sl_p > 0:
                        gap = 0.0
                    elif sl_ratio == 1:
                        gap = self.random_state.random_sample()
                    elif sl_ratio > 1:
                        gap = self.random_state.random_sample()*1.0/sl_ratio
                    elif sl_ratio < 1:
                        gap = (1 - sl_ratio) + \
                            self.random_state.random_sample()*sl_ratio
                    dif = n[atti] - p[atti]
                    s[atti] = p[atti] + gap*dif
                samples.append(s)

        if len(samples) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "No samples generated")
            return X.copy(), y.copy()
        else:
            return (np.vstack([X, np.vstack(samples)]),
                    np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{msmote,
                             author = {Hu, Shengguo and Liang,
                                 Yanfeng and Ma, Lintao and He, Ying},
                             title = {MSMOTE: Improving Classification
                                        Performance When Training Data
                                        is Imbalanced},
                             booktitle = {Proceedings of the 2009 Second
                                            International Workshop on
                                            Computer Science and Engineering
                                            - Volume 02},
                             series = {IWCSE '09},
                             year = {2009},
                             isbn = {978-0-7695-3881-5},
                             pages = {13--17},
                             numpages = {5},
                             url = {https://doi.org/10.1109/WCSE.2009.756},
                             doi = {10.1109/WCSE.2009.756},
                             acmid = {1682710},
                             publisher = {IEEE Computer Society},
                             address = {Washington, DC, USA},
                             keywords = {imbalanced data, over-sampling,
                                        SMOTE, AdaBoost, samples groups,
                                        SMOTEBoost},
                            }

    Notes:
        * The original method was not prepared for the case when all
            minority samples are noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distance, indices = nn.kneighbors(X_min)

        noise_mask = np.repeat(False, len(X_min))

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            index = self.random_state.randint(len(X_min))

            n_p = np.sum(y[indices[index][1:]] == self.min_label)

            if n_p == self.n_neighbors:
                sample_type = 'security'
            elif n_p == 0:
                sample_type = 'noise'
                noise_mask[index] = True
                if np.all(noise_mask):
                    _logger.info("All minority samples are noise")
                    return X.copy(), y.copy()
            else:
                sample_type = 'border'

            if sample_type == 'security':
                neighbor_index = self.random_state.choice(indices[index][1:])
            elif sample_type == 'border':
                neighbor_index = indices[index][1]
            else:
                continue

            s_gen = self.sample_between_points_componentwise(X_min[index],
                                                             X[neighbor_index])
            samples.append(s_gen)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class DE_oversampling(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{de_oversampling,
                            author={Chen, L. and Cai, Z. and Chen, L. and
                                    Gu, Q.},
                            booktitle={2010 Third International Conference
                                       on Knowledge Discovery and Data Mining},
                            title={A Novel Differential Evolution-Clustering
                                    Hybrid Resampling Algorithm on Imbalanced
                                    Datasets},
                            year={2010},
                            volume={},
                            number={},
                            pages={81-85},
                            keywords={pattern clustering;sampling methods;
                                        support vector machines;differential
                                        evolution;clustering algorithm;hybrid
                                        resampling algorithm;imbalanced
                                        datasets;support vector machine;
                                        minority class;mutation operators;
                                        crossover operators;data cleaning
                                        method;F-measure criterion;ROC area
                                        criterion;Support vector machines;
                                        Intrusion detection;Support vector
                                        machine classification;Cleaning;
                                        Electronic mail;Clustering algorithms;
                                        Signal to noise ratio;Learning
                                        systems;Data mining;Geology;imbalanced
                                        datasets;hybrid resampling;clustering;
                                        differential evolution;support vector
                                        machine},
                            doi={10.1109/WKDD.2010.48},
                            ISSN={},
                            month={Jan},}
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 crossover_rate=0.5,
                 similarity_threshold=0.5,
                 n_clusters=30, n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            crossover_rate (float): cross over rate of evoluation
            similarity_threshold (float): similarity threshold paramter
            n_clusters (int): number of clusters for cleansing
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 2)
        self.check_in_range(crossover_rate, 'crossover_rate', [0, 1])
        self.check_in_range(similarity_threshold,
                            'similarity_threshold', [0, 1])
        self.check_greater_or_equal(n_clusters, 'n_clusters', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.crossover_rate = crossover_rate
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'crossover_rate': [0.1, 0.5, 0.9],
                                  'similarity_threshold': [0.5, 0.9],
                                  'n_clusters': [10, 20, 50]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        d = len(X[0])

        X_min = X[y == self.min_label]

        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        distance, indices = nn.kneighbors(X_min)

        # generating samples
        samples = []
        for _ in range(n_to_sample):
            # mutation according to the description in the paper
            random_index = self.random_state.randint(len(X_min))
            random_point = X_min[random_index]
            random_neighbor_indices = self.random_state.choice(
                indices[random_index][1:], 2, replace=False)
            random_neighbor_1 = X_min[random_neighbor_indices[0]]
            random_neighbor_2 = X_min[random_neighbor_indices[1]]

            mutated = random_point + \
                (random_neighbor_1 - random_neighbor_2) * \
                self.random_state.random_sample()

            # crossover - updates the vector 'mutated'
            rand_s = self.random_state.randint(d)
            for i in range(d):
                random_value = self.random_state.random_sample()
                if random_value >= self.crossover_rate and not i == rand_s:
                    mutated[i] = random_point[i]
                elif random_value < self.crossover_rate or i == rand_s:
                    pass

            samples.append(mutated)

        # assembling all data for clearning
        X, y = np.vstack([X, np.vstack(samples)]), np.hstack(
            [y, np.repeat(self.min_label, len(samples))])
        X_min = X[y == self.min_label]

        # cleansing based on clustering
        n_clusters = min([len(X), self.n_clusters])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X)
        unique_labels = np.unique(kmeans.labels_)

        def cluster_filter(li):
            return len(np.unique(y[np.where(kmeans.labels_ == li)[0]])) == 1

        one_label_clusters = [li for li in unique_labels if cluster_filter(li)]
        to_remove = []

        # going through the clusters having one label only
        for li in one_label_clusters:
            cluster_indices = np.where(kmeans.labels_ == li)[0]
            mean_of_cluster = kmeans.cluster_centers_[li]

            # finding center-like sample
            center_like_index = None
            center_like_dist = np.inf

            for i in cluster_indices:
                dist = np.linalg.norm(X[i] - mean_of_cluster)
                if dist < center_like_dist:
                    center_like_dist = dist
                    center_like_index = i

            # removing the samples similar to the center-like sample
            for i in cluster_indices:
                if i != center_like_index:
                    d = np.inner(X[i], X[center_like_index]) / \
                        (np.linalg.norm(X[i]) *
                         np.linalg.norm(X[center_like_index]))
                    if d > self.similarity_threshold:
                        to_remove.append(i)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'crossover_rate': self.crossover_rate,
                'similarity_threshold': self.similarity_threshold,
                'n_clusters': self.n_clusters,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

# Borrowed from sklearn-dev, will be removed once the sklearn implementation
# becomes stable


class OPTICS:
    def __init__(self, min_samples=5, max_eps=np.inf, metric='euclidean',
                 p=2, metric_params=None, maxima_ratio=.75,
                 rejection_ratio=.7, similarity_threshold=0.4,
                 significant_min=.003, min_cluster_size=.005,
                 min_maxima_ratio=0.001, algorithm='ball_tree',
                 leaf_size=30, n_jobs=1):

        self.max_eps = max_eps
        self.min_samples = min_samples
        self.maxima_ratio = maxima_ratio
        self.rejection_ratio = rejection_ratio
        self.similarity_threshold = similarity_threshold
        self.significant_min = significant_min
        self.min_cluster_size = min_cluster_size
        self.min_maxima_ratio = min_maxima_ratio
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Perform OPTICS clustering
        Extracts an ordered list of points and reachability distances, and
        performs initial clustering using `max_eps` distance specified at
        OPTICS object instantiation.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data.
        y : ignored
        Returns
        -------
        self : instance of OPTICS
            The instance.
        """
        n_samples = len(X)

        if self.min_samples > n_samples:
            m = ("Number of training samples (n_samples=%d) must "
                 "be greater than min_samples (min_samples=%d) "
                 "used for clustering.")
            m = m % (n_samples, self.min_samples)
            raise ValueError(self.__class__.__name__ + ": " + m)

        if self.min_cluster_size <= 0 or (self.min_cluster_size !=
                                          int(self.min_cluster_size)
                                          and self.min_cluster_size > 1):
            m = ('min_cluster_size must be a positive integer or '
                 'a float between 0 and 1. Got %r')
            m = m % self.min_cluster_size
            raise ValueError(self.__class__.__name__ + ": " + m)
        elif self.min_cluster_size > n_samples:
            m = ('min_cluster_size must be no greater than the '
                 'number of samples (%d). Got %d')
            m = m % (n_samples, self.min_cluster_size)

            raise ValueError(self.__class__.__name__ + ": " + m)

        # Start all points as 'unprocessed' ##
        self.reachability_ = np.empty(n_samples)
        self.reachability_.fill(np.inf)
        self.core_distances_ = np.empty(n_samples)
        self.core_distances_.fill(np.nan)
        # Start all points as noise ##
        self.labels_ = np.full(n_samples, -1, dtype=int)

        nbrs = NearestNeighbors(n_neighbors=self.min_samples,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size, metric=self.metric,
                                metric_params=self.metric_params, p=self.p,
                                n_jobs=self.n_jobs)

        nbrs.fit(X)
        self.core_distances_[:] = nbrs.kneighbors(X,
                                                  self.min_samples)[0][:, -1]

        self.ordering_ = self._calculate_optics_order(X, nbrs)

        return self

    # OPTICS helper functions

    def _calculate_optics_order(self, X, nbrs):
        # Main OPTICS loop. Not parallelizable. The order that entries are
        # written to the 'ordering_' list is important!
        processed = np.zeros(X.shape[0], dtype=bool)
        ordering = np.zeros(X.shape[0], dtype=int)
        ordering_idx = 0
        for point in range(X.shape[0]):
            if processed[point]:
                continue
            if self.core_distances_[point] <= self.max_eps:
                while not processed[point]:
                    processed[point] = True
                    ordering[ordering_idx] = point
                    ordering_idx += 1
                    point = self._set_reach_dist(point, processed, X, nbrs)
            else:  # For very noisy points
                ordering[ordering_idx] = point
                ordering_idx += 1
                processed[point] = True
        return ordering

    def _set_reach_dist(self, point_index, processed, X, nbrs):
        P = X[point_index:point_index + 1]
        indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                        return_distance=False)[0]

        # Getting indices of neighbors that have not been processed
        unproc = np.compress((~np.take(processed, indices)).ravel(),
                             indices, axis=0)
        # Keep n_jobs = 1 in the following lines...please
        if not unproc.size:
            # Everything is already processed. Return to main loop
            return point_index

        dists = pairwise_distances(P, np.take(X, unproc, axis=0),
                                   self.metric, n_jobs=1).ravel()

        rdists = np.maximum(dists, self.core_distances_[point_index])
        new_reach = np.minimum(np.take(self.reachability_, unproc), rdists)
        self.reachability_[unproc] = new_reach

        # Define return order based on reachability distance
        return (unproc[self.quick_scan(np.take(self.reachability_, unproc),
                                       dists)])

    def isclose(self, a, b, rel_tol=1e-09, abs_tol=0.0):
        return abs(a-b) <= max([rel_tol*max([abs(a), abs(b)]), abs_tol])

    def quick_scan(self, rdists, dists):
        rdist = np.inf
        dist = np.inf
        n = len(rdists)
        for i in range(n):
            if rdists[i] < rdist:
                rdist = rdists[i]
                dist = dists[i]
                idx = i
            elif self.isclose(rdists[i], rdist):
                if dists[i] < dist:
                    dist = dists[i]
                    idx = i
        return idx


class SMOBD(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{smobd,
                            author={Cao, Q. and Wang, S.},
                            booktitle={2011 International Conference on
                                        Information Management, Innovation
                                        Management and Industrial
                                        Engineering},
                            title={Applying Over-sampling Technique Based
                                     on Data Density and Cost-sensitive
                                     SVM to Imbalanced Learning},
                            year={2011},
                            volume={2},
                            number={},
                            pages={543-548},
                            keywords={data handling;learning (artificial
                                        intelligence);support vector machines;
                                        oversampling technique application;
                                        data density;cost sensitive SVM;
                                        imbalanced learning;SMOTE algorithm;
                                        data distribution;density information;
                                        Support vector machines;Classification
                                        algorithms;Noise measurement;Arrays;
                                        Noise;Algorithm design and analysis;
                                        Training;imbalanced learning;
                                        cost-sensitive SVM;SMOTE;data density;
                                        SMOBD},
                            doi={10.1109/ICIII.2011.276},
                            ISSN={2155-1456},
                            month={Nov},}
    """

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal]

    def __init__(self,
                 proportion=1.0,
                 eta1=0.5,
                 t=1.8,
                 min_samples=5,
                 max_eps=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            eta1 (float): control parameter of density estimation
            t (float): control parameter of noise filtering
            min_samples (int): minimum samples parameter for OPTICS
            max_eps (float): maximum environment radius paramter for OPTICS
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_in_range(eta1, 'eta1', [0.0, 1.0])
        self.check_greater_or_equal(t, 't', 0)
        self.check_greater_or_equal(min_samples, 'min_samples', 1)
        self.check_greater_or_equal(max_eps, 'max_eps', 0.0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.eta1 = eta1
        self.t = t
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'eta1': [0.1, 0.5, 0.9],
                                  't': [1.5, 2.5],
                                  'min_samples': [5],
                                  'max_eps': [0.1, 0.5, 1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # running the OPTICS technique based on the sklearn implementation
        # TODO: replace to sklearn call once it is stable
        min_samples = min([len(X_min)-1, self.min_samples])
        o = OPTICS(min_samples=min_samples,
                   max_eps=self.max_eps,
                   n_jobs=self.n_jobs)
        o.fit(X_min)
        cd = o.core_distances_
        rd = o.reachability_

        # noise filtering
        cd_average = np.mean(cd)
        rd_average = np.mean(rd)
        noise = np.logical_and(cd > cd_average*self.t, rd > rd_average*self.t)

        # fitting a nearest neighbor model to be able to find
        # neighbors in radius
        n_neighbors = min([len(X_min), self.min_samples+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        distances, indices = nn.kneighbors(X_min)

        # determining the density
        factor_1 = cd
        factor_2 = np.array([len(x) for x in nn.radius_neighbors(
            X_min, radius=self.max_eps, return_distance=False)])

        if max(factor_1) == 0 or max(factor_2) == 0:
            return X.copy(), y.copy()

        factor_1 = factor_1/max(factor_1)
        factor_2 = factor_2/max(factor_2)

        df = factor_1*self.eta1 + factor_2*(1 - self.eta1)

        # setting the density at noisy samples to zero
        for i in range(len(noise)):
            if noise[i]:
                df[i] = 0

        if sum(df) == 0 or any(np.isnan(df)) or any(np.isinf(df)):
            return X.copy(), y.copy()

        # normalizing the density
        df_dens = df/sum(df)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)), p=df_dens)
            neighbor_idx = self.random_state.choice(indices[idx][1:])
            samples.append(self.sample_between_points_componentwise(
                X_min[idx], X_min[neighbor_idx]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'eta1': self.eta1,
                't': self.t,
                'min_samples': self.min_samples,
                'max_eps': self.max_eps,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SUNDO(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{sundo,
                            author={Cateni, S. and Colla, V. and Vannucci, M.},
                            booktitle={2011 11th International Conference on
                                        Intelligent Systems Design and
                                        Applications},
                            title={Novel resampling method for the
                                    classification of imbalanced datasets for
                                    industrial and other real-world problems},
                            year={2011},
                            volume={},
                            number={},
                            pages={402-407},
                            keywords={decision trees;pattern classification;
                                        sampling methods;support vector
                                        machines;resampling method;imbalanced
                                        dataset classification;industrial
                                        problem;real world problem;
                                        oversampling technique;undersampling
                                        technique;support vector machine;
                                        decision tree;binary classification;
                                        synthetic dataset;public dataset;
                                        industrial dataset;Support vector
                                        machines;Training;Accuracy;Databases;
                                        Intelligent systems;Breast cancer;
                                        Decision trees;oversampling;
                                        undersampling;imbalanced dataset},
                            doi={10.1109/ISDA.2011.6121689},
                            ISSN={2164-7151},
                            month={Nov}}
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_application]

    def __init__(self, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return [{}]

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_1 = len(X_min)
        n_0 = len(X) - n_1
        N = int(np.rint(0.5*n_0 - 0.5*n_1 + 0.5))

        if N == 0:
            return X.copy(), y.copy()

        # generating minority samples
        samples = []

        nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
        nn.fit(X_maj)

        stds = np.std(X_min, axis=0)
        # At one point the algorithm says to keep those points which are
        # the most distant from majority samples, and not leaving any minority
        # sample isolated. This can be implemented by generating multiple
        # samples for each point and keep the one most distant from the
        # majority samples.
        for _ in range(N):
            i = self.random_state.randint(len(X_min))
            best_sample = None
            best_sample_dist = 0
            for _ in range(3):
                s = self.random_state.normal(X_min[i], stds)
                dist, ind = nn.kneighbors(s.reshape(1, -1))
                if dist[0][0] > best_sample_dist:
                    best_sample_dist = dist[0][0]
                    best_sample = s
            samples.append(best_sample)

        # Extending the minority dataset with the new samples
        X_min_extended = np.vstack([X_min, np.vstack(samples)])

        # Removing N elements from the majority dataset

        # normalize
        mms = MinMaxScaler()
        X_maj_normalized = mms.fit_transform(X_maj)

        # computing the distance matrix
        dm = pairwise_distances(X_maj_normalized, X_maj_normalized)

        # len(X_maj) offsets for the diagonal 0 elements, 2N because
        # every distances appears twice
        threshold = sorted(dm.flatten())[min(
            [len(X_maj) + 2*N, len(dm)*len(dm) - 1])]
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # extracting the coordinates of pairs closer than threshold
        pairs_to_break = np.where(dm < threshold)
        pairs_to_break = np.vstack(pairs_to_break)

        # sorting the pairs, otherwise both points would be removed
        pairs_to_break.sort(axis=0)

        # uniqueing the coordinates - the final number might be less than N
        to_remove = np.unique(pairs_to_break[0])

        # removing the selected elements
        X_maj_cleaned = np.delete(X_maj, to_remove, axis=0)

        return (np.vstack([X_min_extended, X_maj_cleaned]),
                np.hstack([np.repeat(self.min_label, len(X_min_extended)),
                           np.repeat(self.maj_label, len(X_maj_cleaned))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MSYN(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{msyn,
                            author="Fan, Xiannian
                            and Tang, Ke
                            and Weise, Thomas",
                            editor="Huang, Joshua Zhexue
                            and Cao, Longbing
                            and Srivastava, Jaideep",
                            title="Margin-Based Over-Sampling Method for
                                    Learning from Imbalanced Datasets",
                            booktitle="Advances in Knowledge Discovery and
                                        Data Mining",
                            year="2011",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="309--320",
                            abstract="Learning from imbalanced datasets has
                                        drawn more and more attentions from
                                        both theoretical and practical aspects.
                                        Over- sampling is a popular and simple
                                        method for imbalanced learning. In this
                                        paper, we show that there is an
                                        inherently potential risk associated
                                        with the over-sampling algorithms in
                                        terms of the large margin principle.
                                        Then we propose a new synthetic over
                                        sampling method, named Margin-guided
                                        Synthetic Over-sampling (MSYN), to
                                        reduce this risk. The MSYN improves
                                        learning with respect to the data
                                        distributions guided by the
                                        margin-based rule. Empirical study
                                        verities the efficacy of MSYN.",
                            isbn="978-3-642-20847-8"
                            }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 pressure=1.5,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            pressure (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(pressure, 'pressure', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.pressure = pressure
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'pressure': [2.5, 2.0, 1.5],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        min_indices = np.where(y == self.min_label)[0]
        maj_indices = np.where(y == self.maj_label)[0]

        # generating samples
        smote = SMOTE(proportion=self.pressure,
                      n_neighbors=self.n_neighbors,
                      n_jobs=self.n_jobs,
                      random_state=self.random_state)

        X_res, y_res = smote.sample(X, y)
        X_new, _ = X_res[len(X):], y_res[len(X):]

        if len(X_new) == 0:
            m = "Sampling is not needed"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # Compute nearest hit and miss for both classes
        nn = NearestNeighbors(n_neighbors=len(X), n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X)

        # computing nearest hit and miss distances, these will be used to
        # compute thetas
        nearest_hit_dist = np.array([dist[i][next(j for j in range(
            1, len(X)) if y[i] == y[ind[i][j]])] for i in range(len(X))])
        nearest_miss_dist = np.array([dist[i][next(j for j in range(
            1, len(X)) if y[i] != y[ind[i][j]])] for i in range(len(X))])

        # computing the thetas without new samples being involved
        theta_A_sub_alpha = 0.5*(nearest_miss_dist - nearest_hit_dist)
        theta_min = theta_A_sub_alpha[min_indices]
        theta_maj = theta_A_sub_alpha[maj_indices]

        # computing the f_3 score for all new samples
        f_3 = []
        for x in X_new:
            # determining the distances of the new sample from the training set
            distances = np.linalg.norm(X - x, axis=1)

            # computing nearest hit and miss distances involving the new
            # elements
            mask = nearest_hit_dist[min_indices] < distances[min_indices]
            nearest_hit_dist_min = np.where(mask,
                                            nearest_hit_dist[min_indices],
                                            distances[min_indices])
            nearest_miss_dist_min = nearest_miss_dist[min_indices]
            nearest_hit_dist_maj = nearest_hit_dist[maj_indices]
            mask = nearest_miss_dist[maj_indices] < distances[maj_indices]
            nearest_miss_dist_maj = np.where(mask,
                                             nearest_miss_dist[maj_indices],
                                             distances[maj_indices])

            # computing the thetas incorporating the new elements
            theta_x_min = 0.5*(nearest_miss_dist_min - nearest_hit_dist_min)
            theta_x_maj = 0.5*(nearest_miss_dist_maj - nearest_hit_dist_maj)

            # determining the delta scores and computing f_3
            Delta_P = np.sum(theta_x_min - theta_min)
            Delta_N = np.sum(theta_x_maj - theta_maj)

            f_3.append(-Delta_N/(Delta_P + 0.01))

        f_3 = np.array(f_3)

        # determining the elements with the minimum f_3 scores to add
        _, new_ind = zip(
            *sorted(zip(f_3, np.arange(len(f_3))), key=lambda x: x[0]))
        new_ind = list(new_ind[:(len(X_maj) - len(X_min))])

        return (np.vstack([X, X_new[new_ind]]),
                np.hstack([y, np.repeat(self.min_label, len(new_ind))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'pressure': self.pressure,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SVM_balance(OverSampling):
    """
    References:
        * BibTex::

            @article{svm_balance,
                     author = {Farquad, M.A.H. and Bose, Indranil},
                     title = {Preprocessing Unbalanced Data Using Support
                                Vector Machine},
                     journal = {Decis. Support Syst.},
                     issue_date = {April, 2012},
                     volume = {53},
                     number = {1},
                     month = apr,
                     year = {2012},
                     issn = {0167-9236},
                     pages = {226--233},
                     numpages = {8},
                     url = {http://dx.doi.org/10.1016/j.dss.2012.01.016},
                     doi = {10.1016/j.dss.2012.01.016},
                     acmid = {2181554},
                     publisher = {Elsevier Science Publishers B. V.},
                     address = {Amsterdam, The Netherlands, The Netherlands},
                     keywords = {COIL data, Hybrid method, Preprocessor, SVM,
                                    Unbalanced data},
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X, y = SMOTE(proportion=self.proportion,
                     n_neighbors=self.n_neighbors,
                     n_jobs=self.n_jobs,
                     random_state=self.random_state).sample(X, y)

        if sum(y == self.min_label) < 2:
            return X.copy(), y.copy()
        else:
            cv = min([5, sum(y == self.min_label)])

        ss = StandardScaler()
        X_norm = ss.fit_transform(X)

        C_params = [0.01, 0.1, 1.0, 10.0]
        best_score = 0
        best_C = 0.01
        for C in C_params:
            _logger.info(self.__class__.__name__ + ": " +
                         "Evaluating SVM with C=%f" % C)
            svc = SVC(C=C, kernel='rbf', gamma='auto')
            score = np.mean(cross_val_score(svc, X_norm, y, cv=cv))
            if score > best_score:
                best_score = score
                best_C = C
        svc = SVC(C=best_C, kernel='rbf', gamma='auto')
        svc.fit(X_norm, y)

        return X, svc.predict(X_norm)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class TRIM_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{trim_smote,
                            author="Puntumapon, Kamthorn
                            and Waiyamai, Kitsana",
                            editor="Tan, Pang-Ning
                            and Chawla, Sanjay
                            and Ho, Chin Kuan
                            and Bailey, James",
                            title="A Pruning-Based Approach for Searching
                                    Precise and Generalized Region for
                                    Synthetic Minority Over-Sampling",
                            booktitle="Advances in Knowledge Discovery
                                        and Data Mining",
                            year="2012",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="371--382",
                            isbn="978-3-642-30220-6"
                            }

    Notes:
        * It is not described precisely how the filtered data is used for
            sample generation. The method is proposed to be a preprocessing
            step, and it states that it applies sample generation to each
            group extracted.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 min_precision=0.3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(min_precision, 'min_precision', [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.min_precision = min_precision
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'min_precision': [0.3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def trim(self, y):
        """
        Determines the trim value.

        Args:
            y (np.array): array of target labels

        Returns:
            float: the trim value
        """
        return np.sum(y == self.min_label)**2/len(y)

    def precision(self, y):
        """
        Determines the precision value.

        Args:
            y (np.array): array of target labels

        Returns:
            float: the precision value
        """
        return np.sum(y == self.min_label)/len(y)

    def determine_splitting_point(self, X, y, split_on_border=False):
        """
        Determines the splitting point.

        Args:
            X (np.matrix): a subset of the training data
            y (np.array): an array of target labels
            split_on_border (bool): wether splitting on class borders is
                                    considered

        Returns:
            tuple(int, float), bool: (splitting feature, splitting value),
                                        make the split
        """
        trim_value = self.trim(y)
        d = len(X[0])
        max_t_minus_gain = 0.0
        split = None

        # checking all dimensions of X
        for i in range(d):
            # sort the elements in dimension i
            sorted_X_y = sorted(zip(X[:, i], y), key=lambda pair: pair[0])
            sorted_y = [yy for _, yy in sorted_X_y]

            # number of minority samples on the left
            left_min = 0
            # number of minority samples on the right
            right_min = np.sum(sorted_y == self.min_label)

            # check all possible splitting points sequentiall
            for j in range(0, len(sorted_y)-1):
                if sorted_y[j] == self.min_label:
                    # adjusting the number of minority and majority samples
                    left_min = left_min + 1
                    right_min = right_min - 1
                # checking of we can split on the border and do not split
                # tieing feature values
                if ((split_on_border is False
                     or (split_on_border is True
                         and not sorted_y[j-1] == sorted_y[j]))
                        and sorted_X_y[j][0] != sorted_X_y[j+1][0]):
                    # compute trim value of the left
                    trim_left = left_min**2/(j+1)
                    # compute trim value of the right
                    trim_right = right_min**2/(len(sorted_y) - j - 1)
                    # let's check the gain
                    if max([trim_left, trim_right]) > max_t_minus_gain:
                        max_t_minus_gain = max([trim_left, trim_right])
                        split = (i, sorted_X_y[j][0])
        # return splitting values and the value of the logical condition
        # in line 9
        if split is not None:
            return split, max_t_minus_gain > trim_value
        else:
            return (0, 0), False

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        leafs = [(X, y)]
        candidates = []
        seeds = []

        # executing the trimming
        # loop in line 2 of the paper
        _logger.info(self.__class__.__name__ +
                     ": " + "do the trimming process")
        while len(leafs) > 0 or len(candidates) > 0:
            add_to_leafs = []
            # executing the loop starting in line 3
            for leaf in leafs:
                # the function implements the loop starting in line 6
                # splitting on class border is forced
                split, gain = self.determine_splitting_point(
                    leaf[0], leaf[1], True)
                if len(leaf[0]) == 1:
                    # small leafs with 1 element (no splitting point)
                    # are dropped as noise
                    continue
                else:
                    # condition in line 9
                    if gain:
                        # making the split
                        mask_left = (leaf[0][:, split[0]] <= split[1])
                        X_left = leaf[0][mask_left]
                        y_left = leaf[1][mask_left]
                        mask_right = np.logical_not(mask_left)
                        X_right = leaf[0][mask_right]
                        y_right = leaf[1][mask_right]

                        # condition in line 11
                        if np.sum(y_left == self.min_label) > 0:
                            add_to_leafs.append((X_left, y_left))
                        # condition in line 13
                        if np.sum(y_right == self.min_label) > 0:
                            add_to_leafs.append((X_right, y_right))
                    else:
                        # line 16
                        candidates.append(leaf)
            # we implement line 15 and 18 by replacing the list of leafs by
            # the list of new leafs.
            leafs = add_to_leafs

            # iterating through all candidates (loop starting in line 21)
            for c in candidates:
                # extracting splitting points, this time split on border
                # is not forced
                split, gain = self.determine_splitting_point(c[0], c[1], False)
                if len(c[0]) == 1:
                    # small leafs are dropped as noise
                    continue
                else:
                    # checking condition in line 27
                    if gain:
                        # doing the split
                        mask_left = (c[0][:, split[0]] <= split[1])
                        X_left, y_left = c[0][mask_left], c[1][mask_left]
                        mask_right = np.logical_not(mask_left)
                        X_right, y_right = c[0][mask_right], c[1][mask_right]
                        # checking logic in line 29
                        if np.sum(y_left == self.min_label) > 0:
                            leafs.append((X_left, y_left))
                        # checking logic in line 31
                        if np.sum(y_right == self.min_label) > 0:
                            leafs.append((X_right, y_right))
                    else:
                        # adding candidate to seeds (line 35)
                        seeds.append(c)
            # line 33 and line 36 are implemented by emptying the candidates
            # list
            candidates = []

        # filtering the resulting set
        filtered_seeds = [s for s in seeds if self.precision(
            s[1]) > self.min_precision]

        # handling the situation when no seeds were found
        if len(seeds) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "no seeds identified")
            return X.copy(), y.copy()

        # fix for bad choice of min_precision
        multiplier = 0.9
        while len(filtered_seeds) == 0:
            filtered_seeds = [s for s in seeds if self.precision(
                s[1]) > self.min_precision*multiplier]
            multiplier = multiplier*0.9
            if multiplier < 0.1:
                _logger.warning(self.__class__.__name__ + ": " +
                                "no clusters passing the filtering")
                return X.copy(), y.copy()

        seeds = filtered_seeds

        X_seed = np.vstack([s[0] for s in seeds])
        y_seed = np.hstack([s[1] for s in seeds])

        _logger.info(self.__class__.__name__ + ": " + "do the sampling")
        # generating samples by SMOTE
        X_seed_min = X_seed[y_seed == self.min_label]
        if len(X_seed_min) <= 1:
            _logger.warning(self.__class__.__name__ + ": " +
                            "X_seed_min contains less than 2 samples")
            return X.copy(), y.copy()

        n_neighbors = min([len(X_seed_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_seed_min)
        distances, indices = nn.kneighbors(X_seed_min)

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.randint(len(X_seed_min))
            random_neighbor_idx = self.random_state.choice(
                indices[random_idx][1:])
            samples.append(self.sample_between_points(
                X_seed_min[random_idx], X_seed_min[random_neighbor_idx]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'min_precision': self.min_precision,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_RSB(OverSampling):
    """
    References:
        * BibTex::

            @Article{smote_rsb,
                    author="Ramentol, Enislay
                    and Caballero, Yail{\'e}
                    and Bello, Rafael
                    and Herrera, Francisco",
                    title="SMOTE-RSB*: a hybrid preprocessing approach
                            based on oversampling and undersampling for
                            high imbalanced data-sets using SMOTE and
                            rough sets theory",
                    journal="Knowledge and Information Systems",
                    year="2012",
                    month="Nov",
                    day="01",
                    volume="33",
                    number="2",
                    pages="245--265",
                    issn="0219-3116",
                    doi="10.1007/s10115-011-0465-6",
                    url="https://doi.org/10.1007/s10115-011-0465-6"
                    }

    Notes:
        * I think the description of the algorithm in Fig 5 of the paper
            is not correct. The set "resultSet" is initialized with the
            original instances, and then the While loop in the Algorithm
            run until resultSet is empty, which never holds. Also, the
            resultSet is only extended in the loop. Our implementation
            is changed in the following way: we generate twice as many
            instances are required to balance the dataset, and repeat
            the loop until the number of new samples added to the training
            set is enough to balance the dataset.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=2.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X_maj = X[y == self.maj_label]
        X_min = X[y == self.min_label]

        # Step 1: do the sampling
        smote = SMOTE(proportion=self.proportion,
                      n_neighbors=self.n_neighbors,
                      n_jobs=self.n_jobs,
                      random_state=self.random_state)

        X_samp, y_samp = smote.sample(X, y)
        X_samp, y_samp = X_samp[len(X):], y_samp[len(X):]

        if len(X_samp) == 0:
            return X.copy(), y.copy()

        # Step 2: (original will be added later)
        result_set = []

        # Step 3: first the data is normalized
        maximums = np.max(X_samp, axis=0)
        minimums = np.min(X_samp, axis=0)

        # normalize X_new and X_maj
        norm_factor = maximums - minimums
        null_mask = norm_factor == 0
        n_null = np.sum(null_mask)
        fixed = np.max(np.vstack([maximums[null_mask], np.repeat(1, n_null)]),
                       axis=0)

        norm_factor[null_mask] = fixed

        X_samp_norm = X_samp / norm_factor
        X_maj_norm = X_maj / norm_factor

        # compute similarity matrix
        similarity_matrix = 1.0 - pairwise_distances(X_samp_norm,
                                                     X_maj_norm,
                                                     metric='minkowski',
                                                     p=1)/len(X[0])

        # Step 4: counting the similar examples
        similarity_value = 0.4
        syn = len(X_samp)
        cont = np.zeros(syn)

        already_added = np.repeat(False, len(X_samp))

        while (len(result_set) < len(X_maj) - len(X_min)
                and similarity_value <= 0.9):
            for i in range(syn):
                cont[i] = np.sum(similarity_matrix[i, :] > similarity_value)
                if cont[i] == 0 and not already_added[i]:
                    result_set.append(X_samp[i])
                    already_added[i] = True
            similarity_value = similarity_value + 0.05

        # Step 5: returning the results depending the number of instances
        # added to the result set
        if len(result_set) > 0:
            return (np.vstack([X, np.vstack(result_set)]),
                    np.hstack([y, np.repeat(self.min_label,
                                            len(result_set))]))
        else:
            return np.vstack([X, X_samp]), np.hstack([y, y_samp])

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ProWSyn(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{prowsyn,
                        author="Barua, Sukarna
                        and Islam, Md. Monirul
                        and Murase, Kazuyuki",
                        editor="Pei, Jian
                        and Tseng, Vincent S.
                        and Cao, Longbing
                        and Motoda, Hiroshi
                        and Xu, Guandong",
                        title="ProWSyn: Proximity Weighted Synthetic
                                        Oversampling Technique for
                                        Imbalanced Data Set Learning",
                        booktitle="Advances in Knowledge Discovery
                                    and Data Mining",
                        year="2013",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="317--328",
                        isbn="978-3-642-37456-2"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 L=5,
                 theta=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            L (int): number of levels
            theta (float): smoothing factor in weight formula
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(L, "L", 1)
        self.check_greater_or_equal(theta, "theta", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.L = L
        self.theta = theta
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'L': [3, 5, 7],
                                  'theta': [0.1, 1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and
                                    target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # Step 1 - a bit generalized
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            m = "Sampling is not needed"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # Step 2
        P = np.where(y == self.min_label)[0]
        X_maj = X[y == self.maj_label]

        Ps = []
        proximity_levels = []

        # Step 3
        for i in range(self.L):
            if len(P) == 0:
                break
            # Step 3 a
            n_neighbors = min([len(P), self.n_neighbors])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            nn.fit(X[P])
            distances, indices = nn.kneighbors(X_maj)

            # Step 3 b
            P_i = np.unique(np.hstack([i for i in indices]))

            # Step 3 c - proximity levels are encoded in the Ps list index
            Ps.append(P[P_i])
            proximity_levels.append(i+1)

            # Step 3 d
            P = np.delete(P, P_i)

        # Step 4
        if len(P) > 0:
            Ps.append(P)

        # Step 5
        if len(P) > 0:
            proximity_levels.append(i)
            proximity_levels = np.array(proximity_levels)

        # Step 6
        weights = np.array([np.exp(-self.theta*(proximity_levels[i] - 1))
                            for i in range(len(proximity_levels))])
        # weights is the probability distribution of sampling in the
        # clusters identified
        weights = weights/np.sum(weights)

        suitable = False
        for i in range(len(weights)):
            if weights[i] > 0 and len(Ps[i]) > 1:
                suitable = True

        if not suitable:
            return X.copy(), y.copy()

        # do the sampling, from each cluster proportionally to the distribution
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(
                np.arange(len(weights)), p=weights)
            if len(Ps[cluster_idx]) > 1:
                random_idx1, random_idx2 = self.random_state.choice(
                    Ps[cluster_idx], 2, replace=False)
                samples.append(self.sample_between_points(
                    X[random_idx1], X[random_idx2]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'L': self.L,
                'theta': self.theta,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SL_graph_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{sl_graph_smote,
                    author = {Bunkhumpornpat,
                        Chumpol and Subpaiboonkit, Sitthichoke},
                    booktitle= {13th International Symposium on Communications
                                and Information Technologies},
                    year = {2013},
                    month = {09},
                    pages = {570-575},
                    title = {Safe level graph for synthetic minority
                                over-sampling techniques},
                    isbn = {978-1-4673-5578-0}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # Fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X[y == self.min_label])

        # Computing safe level values
        safe_level_values = np.array(
            [np.sum(y[i] == self.min_label) for i in indices])

        # Computing skewness
        skewness = skew(safe_level_values)

        if skewness < 0:
            # left skewed
            s = Safe_Level_SMOTE(self.proportion,
                                 self.n_neighbors,
                                 n_jobs=self.n_jobs,
                                 random_state=self.random_state)
        else:
            # right skewed
            s = Borderline_SMOTE1(self.proportion,
                                  self.n_neighbors,
                                  n_jobs=self.n_jobs,
                                  random_state=self.random_state)

        return s.sample(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class NRSBoundary_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{nrsboundary_smote,
                    author= {Feng, Hu and Hang, Li},
                    title= {A Novel Boundary Oversampling Algorithm Based on
                            Neighborhood Rough Set Model: NRSBoundary-SMOTE},
                    journal= {Mathematical Problems in Engineering},
                    year= {2013},
                    pages= {10},
                    doi= {10.1155/2013/694809},
                    url= {http://dx.doi.org/10.1155/694809}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 w=0.005,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            w (float): used to set neighborhood radius
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.w = w
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'w': [0.005, 0.01, 0.05]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # step 1
        bound_set = []
        pos_set = []

        # step 2
        X_min_indices = np.where(y == self.min_label)[0]
        X_min = X[X_min_indices]

        # step 3
        dm = pairwise_distances(X, X)
        d_max = np.max(dm, axis=1)
        max_dist = np.max(dm)
        np.fill_diagonal(dm, max_dist)
        d_min = np.min(dm, axis=1)

        delta = d_min + self.w*(d_max - d_min)

        # number of neighbors is not interesting here, as we use the
        # radius_neighbors function to extract the neighbors in a given radius
        n_neighbors = min([self.n_neighbors + 1, len(X)])
        nn = NearestNeighbors(n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        for i in range(len(X)):
            indices = nn.radius_neighbors(X[i].reshape(1, -1),
                                          delta[i],
                                          return_distance=False)

            n_minority = np.sum(y[indices[0]] == self.min_label)
            n_majority = np.sum(y[indices[0]] == self.maj_label)
            if y[i] == self.min_label and not n_minority == len(indices[0]):
                bound_set.append(i)
            elif y[i] == self.maj_label and n_majority == len(indices[0]):
                pos_set.append(i)

        bound_set = np.array(bound_set)
        pos_set = np.array(pos_set)

        if len(pos_set) == 0 or len(bound_set) == 0:
            return X.copy(), y.copy()

        # step 4 and 5
        # computing the nearest neighbors of the bound set from the
        # minority set
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        distances, indices = nn.kneighbors(X[bound_set])

        # do the sampling
        samples = []
        trials = 0
        w = self.w
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(len(bound_set))
            random_neighbor_idx = self.random_state.choice(indices[idx][1:])
            x_new = self.sample_between_points(
                X[bound_set[idx]], X_min[random_neighbor_idx])

            # checking the conflict
            dist_from_pos_set = np.linalg.norm(X[pos_set] - x_new, axis=1)
            if np.all(dist_from_pos_set > delta[pos_set]):
                # no conflict
                samples.append(x_new)
            trials = trials + 1
            if trials > 1000 and len(samples) == 0:
                trials = 0
                w = w*0.9

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'w': self.w,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class LVQ_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{lvq_smote,
                              title={LVQ-SMOTE – Learning Vector Quantization
                                    based Synthetic Minority Over–sampling
                                    Technique for biomedical data},
                              author={Munehiro Nakamura and Yusuke Kajiwara
                                     and Atsushi Otsuka and Haruhiko Kimura},
                              booktitle={BioData Mining},
                              year={2013}
                            }

    Notes:
        * This implementation is only a rough approximation of the method
            described in the paper. The main problem is that the paper uses
            many datasets to find similar patterns in the codebooks and
            replicate patterns appearing in other datasets to the imbalanced
            datasets based on their relative position compared to the codebook
            elements. What we do is clustering the minority class to extract
            a codebook as kmeans cluster means, then, find pairs of codebook
            elements which have the most similar relative position to a
            randomly selected pair of codebook elements, and translate nearby
            minority samples from the neighborhood one pair of codebook
            elements to the neighborood of another pair of codebook elements.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_application]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_clusters=10,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            n_clusters (int): number of clusters in vector quantization
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_clusters", 3)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_clusters': [4, 8, 12]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # clustering X_min to extract codebook
        n_clusters = min([len(X_min), self.n_clusters])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X_min)
        codebook = kmeans.cluster_centers_

        # get nearest neighbors of minority samples to codebook samples
        n_neighbors = min([len(X_min), self.n_neighbors])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        distances, indices = nn.kneighbors(codebook)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # randomly selecting a pair of codebook elements
            cb_0, cb_1 = self.random_state.choice(
                list(range(len(codebook))), 2, replace=False)
            diff = codebook[cb_0] - codebook[cb_1]
            min_dist = np.inf
            min_0 = None
            # finding another pair of codebook elements with similar offset
            for i in range(len(codebook)):
                for j in range(len(codebook)):
                    if cb_0 != i and cb_0 != j and cb_1 != i and cb_1 != j:
                        dd = np.linalg.norm(diff - (codebook[i] - codebook[j]))
                        if dd < min_dist:
                            min_dist = dd
                            min_0 = self.random_state.choice([i, j])

            # translating a random neighbor of codebook element min_0 to
            # the neighborhood of point_0
            random_index = self.random_state.randint(len(indices[min_0]))
            sample = X_min[indices[min_0][random_index]]
            point_0 = codebook[cb_0] + (sample - codebook[min_0])

            samples.append(point_0)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_clusters': self.n_clusters,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SOI_CJ(OverSampling):
    """
    References:
        * BibTex::

            @article{soi_cj,
                    author = {Sánchez, Atlántida I. and Morales, Eduardo and
                                Gonzalez, Jesus},
                    year = {2013},
                    month = {01},
                    pages = {},
                    title = {Synthetic Oversampling of Instances Using
                                Clustering},
                    volume = {22},
                    booktitle = {International Journal of Artificial
                                    Intelligence Tools}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_componentwise]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 method='interpolation',
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of nearest neighbors in the SMOTE
                                sampling
            method (str): 'interpolation'/'jittering'
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_isin(method, 'method', ['interpolation', 'jittering'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.method = method
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'method': ['interpolation', 'jittering']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def clustering(self, X, y):
        """
        Implementation of the clustering technique described in the paper.

        Args:
            X (np.matrix): array of training instances
            y (np.array): target labels

        Returns:
            list(set): list of minority clusters
        """
        nn_all = NearestNeighbors(n_jobs=self.n_jobs)
        nn_all.fit(X)

        X_min = X[y == self.min_label]

        # extract nearest neighbors of all samples from the set of
        # minority samples
        nn = NearestNeighbors(n_neighbors=len(X_min), n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_min)

        # initialize clusters by minority samples
        clusters = []
        for i in range(len(X_min)):
            # empty cluster added
            clusters.append(set())
            # while the closest instance is from the minority class, adding it
            # to the cluster
            for j in indices[i]:
                if y[j] == self.min_label:
                    clusters[i].add(j)
                else:
                    break

        # cluster merging phase
        is_intersection = True
        while is_intersection:
            is_intersection = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # computing intersection
                    intersection = clusters[i].intersection(clusters[j])
                    if len(intersection) > 0:
                        is_intersection = True
                        # computing distance matrix
                        dm = pairwise_distances(
                            X[list(clusters[i])], X[list(clusters[j])])
                        # largest distance
                        max_dist_pair = np.where(dm == np.max(dm))
                        # elements with the largest distance
                        max_i = X[list(clusters[i])[max_dist_pair[0][0]]]
                        max_j = X[list(clusters[j])[max_dist_pair[1][0]]]

                        # finding midpoint and radius
                        mid_point = (max_i + max_j)/2.0
                        radius = np.linalg.norm(mid_point - max_i)

                        # extracting points within the hypersphare of
                        # radius "radius"
                        mid_point_reshaped = mid_point.reshape(1, -1)
                        ind = nn_all.radius_neighbors(mid_point_reshaped,
                                                      radius,
                                                      return_distance=False)

                        n_min = np.sum(y[ind[0]] == self.min_label)
                        if n_min > len(ind[0])/2:
                            # if most of the covered elements come from the
                            # minority class, merge clusters
                            clusters[i].update(clusters[j])
                            clusters[j] = set()
                        else:
                            # otherwise move the difference to the
                            # bigger cluster
                            if len(clusters[i]) > len(clusters[j]):
                                clusters[j].difference_update(intersection)
                            else:
                                clusters[i].difference_update(intersection)

        # returning non-empty clusters
        return [c for c in clusters if len(c) > 0]

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        std_min = np.std(X_min, axis=0)

        # do the clustering
        _logger.info(self.__class__.__name__ + ": " + "Executing clustering")
        clusters = self.clustering(X, y)

        # filtering the clusters, at least two points in a cluster are needed
        # for both interpolation and jittering (due to the standard deviation)
        clusters_filtered = [list(c) for c in clusters if len(c) > 2]

        if len(clusters_filtered) > 0:
            # if there are clusters having at least 2 elements, do the sampling
            cluster_nums = [len(c) for c in clusters_filtered]
            cluster_weights = cluster_nums/np.sum(cluster_nums)
            cluster_stds = [np.std(X[clusters_filtered[i]], axis=0)
                            for i in range(len(clusters_filtered))]

            _logger.info(self.__class__.__name__ + ": " +
                         "Executing sample generation")
            samples = []
            while len(samples) < n_to_sample:
                cluster_idx = self.random_state.choice(
                    np.arange(len(clusters_filtered)), p=cluster_weights)
                if self.method == 'interpolation':
                    clust = clusters_filtered[cluster_idx]
                    idx_0, idx_1 = self.random_state.choice(clust,
                                                            2,
                                                            replace=False)
                    X_0, X_1 = X[idx_0], X[idx_1]
                    samples.append(
                        self.sample_between_points_componentwise(X_0, X_1))
                elif self.method == 'jittering':
                    clust_std = cluster_stds[cluster_idx]
                    std = np.min(np.vstack([std_min, clust_std]), axis=0)
                    clust = clusters_filtered[cluster_idx]
                    idx = self.random_state.choice(clust)
                    X_samp = self.sample_by_jittering_componentwise(X[idx],
                                                                    std)
                    samples.append(X_samp)

            return (np.vstack([X, samples]),
                    np.hstack([y, np.array([self.min_label]*len(samples))]))
        else:
            # otherwise fall back to standard smote
            _logger.warning(self.__class__.__name__ + ": " +
                            "No clusters with more than 2 elements")
            return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'method': self.method,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ROSE(OverSampling):
    """
    References:
        * BibTex::

            @Article{rose,
                    author="Menardi, Giovanna
                    and Torelli, Nicola",
                    title="Training and assessing classification rules with
                            imbalanced data",
                    journal="Data Mining and Knowledge Discovery",
                    year="2014",
                    month="Jan",
                    day="01",
                    volume="28",
                    number="1",
                    pages="92--122",
                    issn="1573-756X",
                    doi="10.1007/s10618-012-0295-5",
                    url="https://doi.org/10.1007/s10618-012-0295-5"
                    }

    Notes:
        * It is not entirely clear if the authors propose kernel density
            estimation or the fitting of simple multivariate Gaussians
            on the minority samples. The latter seems to be more likely,
            I implement that approach.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self, proportion=1.0, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0.0)

        self.proportion = proportion

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # Estimating the H matrix
        std = np.std(X_min, axis=0)
        d = len(X[0])
        n = len(X_min)
        H = std*(4.0/((d + 1)*n))**(1.0/(d + 4))

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.randint(len(X_min))
            samples.append(self.sample_by_gaussian_jittering(
                X_min[random_idx], H))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'random_state': self._random_state_init}


class SMOTE_OUT(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                      title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An
                                enhancement strategy to handle imbalance in
                                data level},
                      author={Fajri Koto},
                      journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                      year={2014},
                      pages={280-284}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        # nearest neighbors among minority points
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn_min = NearestNeighbors(n_neighbors=n_neighbors,
                                  n_jobs=self.n_jobs).fit(X_min)

        min_distances, min_indices = nn_min.kneighbors(X_min)
        # nearest neighbors among majority points
        n_neighbors = min([len(X_maj), self.n_neighbors+1])
        nn_maj = NearestNeighbors(
            n_neighbors=n_neighbors, n_jobs=self.n_jobs).fit(X_maj)
        maj_distances, maj_indices = nn_maj.kneighbors(X_min)

        # generate samples
        samples = []
        for _ in range(n_to_sample):
            # implementation of Algorithm 1 in the paper
            random_idx = self.random_state.choice(
                np.arange(len(minority_indices)))
            u = X[minority_indices[random_idx]]
            v = X_maj[self.random_state.choice(maj_indices[random_idx])]
            dif1 = u - v
            uu = u + self.random_state.random_sample()*0.3*dif1
            x = X_min[self.random_state.choice(min_indices[random_idx][1:])]
            dif2 = uu - x
            w = x + self.random_state.random_sample()*0.5*dif2

            samples.append(w)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_Cosine(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                      title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE:
                                An enhancement strategy to handle imbalance
                                in data level},
                      author={Fajri Koto},
                      journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                      year={2014},
                      pages={280-284}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        # Fitting the nearest neighbors models to the minority and
        # majority data using two different metrics for the minority
        nn_min_euc = NearestNeighbors(n_neighbors=len(X_min),
                                      n_jobs=self.n_jobs)
        nn_min_euc.fit(X_min)
        nn_min_euc_dist, nn_min_euc_ind = nn_min_euc.kneighbors(X_min)

        nn_min_cos = NearestNeighbors(n_neighbors=len(X_min),
                                      metric='cosine',
                                      n_jobs=self.n_jobs)
        nn_min_cos.fit(X_min)
        nn_min_cos_dist, nn_min_cos_ind = nn_min_cos.kneighbors(X_min)

        nn_maj = NearestNeighbors(n_neighbors=self.n_neighbors,
                                  n_jobs=self.n_jobs)
        nn_maj.fit(X_maj)
        nn_maj_dist, nn_maj_ind = nn_maj.kneighbors(X_min)

        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.choice(
                np.arange(len(minority_indices)))
            u = X[minority_indices[random_idx]]
            # get the rank of each minority sample according to their distance
            # from u
            to_sort_euc = zip(
                nn_min_euc_ind[random_idx], np.arange(len(X_min)))
            _, sorted_by_euc_ind = zip(*(sorted(to_sort_euc,
                                                key=lambda x: x[0])))
            to_sort_cos = zip(
                nn_min_cos_ind[random_idx], np.arange(len(X_min)))
            _, sorted_by_cos_ind = zip(*(sorted(to_sort_cos,
                                                key=lambda x: x[0])))
            # adding the ranks to get the composite similarity measure (called
            # voting in the paper)
            ranked_min_indices = sorted_by_euc_ind + sorted_by_cos_ind
            # sorting the ranking
            to_sort = zip(ranked_min_indices, np.arange(len(X_min)))
            _, sorted_ranking = zip(*(sorted(to_sort, key=lambda x: x[0])))
            # get the indices of the n_neighbors nearest neighbors according
            # to the composite metrics
            min_indices = sorted_ranking[1:(self.n_neighbors + 1)]

            v = X_maj[self.random_state.choice(nn_maj_ind[random_idx])]
            dif1 = u - v
            uu = u + self.random_state.random_sample()*0.3*dif1
            x = X_min[self.random_state.choice(min_indices[1:])]
            dif2 = uu - x
            w = x + self.random_state.random_sample()*0.5*dif2
            samples.append(w)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Selected_SMOTE(OverSampling):
    """
    References:
        * BibTex::

        @article{smote_out_smote_cosine_selected_smote,
                  title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An
                            enhancement strategy to handle imbalance in
                            data level},
                  author={Fajri Koto},
                  journal={2014 International Conference on Advanced
                            Computer Science and Information System},
                  year={2014},
                  pages={280-284}
                }

    Notes:
        * Significant attribute selection was not described in the paper,
            therefore we have implemented something meaningful.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 perc_sign_attr=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            strategy (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            perc_sign_attr (float): [0,1] - percentage of significant
                                            attributes
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(perc_sign_attr, 'perc_sign_attr', [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.perc_sign_attr = perc_sign_attr
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'perc_sign_attr': [0.3, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn_min_euc = NearestNeighbors(n_neighbors=n_neighbors,
                                      n_jobs=self.n_jobs).fit(X_min)

        nn_min_dist, nn_min_ind = nn_min_euc.kneighbors(X_min)

        # significant attribute selection was not described in the paper
        # I have implemented significant attribute selection by checking
        # the overlap between ranges of minority and majority class attributes
        # the attributes with bigger overlap respecting their ranges
        # are considered more significant
        min_ranges_a = np.min(X_min, axis=0)
        min_ranges_b = np.max(X_min, axis=0)
        maj_ranges_a = np.min(X_maj, axis=0)
        maj_ranges_b = np.max(X_maj, axis=0)

        # end points of overlaps
        max_a = np.max(np.vstack([min_ranges_a, maj_ranges_a]), axis=0)
        min_b = np.min(np.vstack([min_ranges_b, maj_ranges_b]), axis=0)

        # size of overlap
        overlap = min_b - max_a

        # replacing negative values (no overlap) by zero
        overlap = np.where(overlap < 0, 0, overlap)
        # percentage of overlap compared to the ranges of attributes in the
        # minority set
        percentages = overlap/(min_ranges_b - min_ranges_a)
        # fixing zero division if some attributes have zero range
        percentages = np.nan_to_num(percentages)
        # number of significant attributes to determine
        num_sign_attr = min(
            [1, int(np.rint(self.perc_sign_attr*len(percentages)))])

        significant_attr = (percentages >= sorted(
            percentages)[-num_sign_attr]).astype(int)

        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.choice(range(len(minority_indices)))
            u = X[minority_indices[random_idx]]
            v = X_min[self.random_state.choice(nn_min_ind[random_idx][1:])]
            samples.append(self.sample_between_points_componentwise(
                u, v, significant_attr))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'perc_sign_attr': self.perc_sign_attr,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class LN_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ln_smote,
                            author={Maciejewski, T. and Stefanowski, J.},
                            booktitle={2011 IEEE Symposium on Computational
                                        Intelligence and Data Mining (CIDM)},
                            title={Local neighbourhood extension of SMOTE for
                                        mining imbalanced data},
                            year={2011},
                            volume={},
                            number={},
                            pages={104-111},
                            keywords={Bayes methods;data mining;pattern
                                        classification;local neighbourhood
                                        extension;imbalanced data mining;
                                        focused resampling technique;SMOTE
                                        over-sampling method;naive Bayes
                                        classifiers;Noise measurement;Noise;
                                        Decision trees;Breast cancer;
                                        Sensitivity;Data mining;Training},
                            doi={10.1109/CIDM.2011.5949434},
                            ISSN={},
                            month={April}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        if self.n_neighbors + 2 > len(X):
            n_neighbors = len(X) - 2
        else:
            n_neighbors = self.n_neighbors

        if n_neighbors < 2:
            return X.copy(), y.copy()

        # nearest neighbors of each instance to each instance in the dataset
        nn = NearestNeighbors(n_neighbors=n_neighbors + 2, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        minority_indices = np.where(y == self.min_label)[0]

        # dimensionality
        d = len(X[0])

        def safe_level(p_idx, n_idx=None):
            """
            computing the safe level of samples

            Args:
                p_idx (int): index of positive sample
                n_idx (int): index of other sample

            Returns:
                int: safe level
            """
            if n_idx is None:
                # implementation for 1 sample only
                return np.sum(y[indices[p_idx][1:-1]] == self.min_label)
            else:
                # implementation for 2 samples
                if ((not y[n_idx] != self.maj_label)
                        and p_idx in indices[n_idx][1:-1]):
                    # -1 because p_idx will be replaced
                    n_positives = np.sum(
                        y[indices[n_idx][1:-1]] == self.min_label) - 1
                    if y[indices[n_idx][-1]] == self.min_label:
                        # this is the effect of replacing p_idx by the next
                        # (k+1)th neighbor
                        n_positives = n_positives + 1
                    return n_positives
                return np.sum(y[indices[n_idx][1:-1]] == self.min_label)

        def random_gap(slp, sln, n_label):
            """
            determining random gap

            Args:
                slp (int): safe level of p
                sln (int): safe level of n
                n_label (int): label of n

            Returns:
                float: gap
            """
            delta = 0
            if sln == 0 and slp > 0:
                return delta
            else:
                sl_ratio = slp/sln
                if sl_ratio == 1:
                    delta = self.random_state.random_sample()
                elif sl_ratio > 1:
                    delta = self.random_state.random_sample()/sl_ratio
                else:
                    delta = 1.0 - self.random_state.random_sample()*sl_ratio
            if not n_label == self.min_label:
                delta = delta*sln/(n_neighbors)
            return delta

        # generating samples
        trials = 0
        samples = []
        while len(samples) < n_to_sample:
            p_idx = self.random_state.choice(minority_indices)
            # extract random neighbor of p
            n_idx = self.random_state.choice(indices[p_idx][1:-1])

            # checking can-create criteria
            slp = safe_level(p_idx)
            sln = safe_level(p_idx, n_idx)

            if (not slp == 0) or (not sln == 0):
                # can create
                p = X[p_idx]
                n = X[n_idx]
                x_new = p.copy()

                for a in range(d):
                    delta = random_gap(slp, sln, y[n_idx])
                    diff = n[a] - p[a]
                    x_new[a] = p[a] + delta*diff
                samples.append(x_new)

            trials = trials + 1
            if len(samples)/trials < 1.0/n_to_sample:
                _logger.info(self.__class__.__name__ + ": " +
                             "no instances with slp > 0 and sln > 0 found")
                return X.copy(), y.copy()

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MWMOTE(OverSampling):
    """
    References:
        * BibTex::

            @ARTICLE{mwmote,
                        author={Barua, S. and Islam, M. M. and Yao, X. and
                                Murase, K.},
                        journal={IEEE Transactions on Knowledge and Data
                                Engineering},
                        title={MWMOTE--Majority Weighted Minority Oversampling
                                Technique for Imbalanced Data Set Learning},
                        year={2014},
                        volume={26},
                        number={2},
                        pages={405-425},
                        keywords={learning (artificial intelligence);pattern
                                    clustering;sampling methods;AUC;area under
                                    curve;ROC;receiver operating curve;G-mean;
                                    geometric mean;minority class cluster;
                                    clustering approach;weighted informative
                                    minority class samples;Euclidean distance;
                                    hard-to-learn informative minority class
                                    samples;majority class;synthetic minority
                                    class samples;synthetic oversampling
                                    methods;imbalanced learning problems;
                                    imbalanced data set learning;
                                    MWMOTE-majority weighted minority
                                    oversampling technique;Sampling methods;
                                    Noise measurement;Boosting;Simulation;
                                    Complexity theory;Interpolation;Abstracts;
                                    Imbalanced learning;undersampling;
                                    oversampling;synthetic sample generation;
                                    clustering},
                        doi={10.1109/TKDE.2012.232},
                        ISSN={1041-4347},
                        month={Feb}}

    Notes:
        * The original method was not prepared for the case of having clusters
            of 1 elements.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 k1=5,
                 k2=5,
                 k3=5,
                 M=10,
                 cf_th=5.0,
                 cmax=10.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            k1 (int): parameter of the NearestNeighbors component
            k2 (int): parameter of the NearestNeighbors component
            k3 (int): parameter of the NearestNeighbors component
            M (int): number of clusters
            cf_th (float): cutoff threshold
            cmax (float): maximum closeness value
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(k1, 'k1', 1)
        self.check_greater_or_equal(k2, 'k2', 1)
        self.check_greater_or_equal(k3, 'k3', 1)
        self.check_greater_or_equal(M, 'M', 1)
        self.check_greater_or_equal(cf_th, 'cf_th', 0)
        self.check_greater_or_equal(cmax, 'cmax', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.M = M
        self.cf_th = cf_th
        self.cmax = cmax
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k1': [5, 9],
                                  'k2': [5, 9],
                                  'k3': [5, 9],
                                  'M': [4, 10],
                                  'cf_th': [5.0],
                                  'cmax': [10.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority = np.where(y == self.min_label)[0]

        # Step 1
        n_neighbors = min([len(X), self.k1 + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs)
        nn.fit(X)
        dist1, ind1 = nn.kneighbors(X)

        # Step 2
        arr = [i for i in minority if np.sum(y[ind1[i][1:]] == self.min_label)]
        filtered_minority = np.array(arr)

        if len(filtered_minority) == 0:
            _logger.info(self.__class__.__name__ + ": " +
                         "filtered_minority array is empty")
            return X.copy(), y.copy()

        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        nn_maj = NearestNeighbors(n_neighbors=self.k2, n_jobs=self.n_jobs)
        nn_maj.fit(X_maj)
        dist2, ind2 = nn_maj.kneighbors(X[filtered_minority])

        # Step 4
        border_majority = np.unique(ind2.flatten())

        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        n_neighbors = min([self.k3, len(X_min)])
        nn_min = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn_min.fit(X_min)
        dist3, ind3 = nn_min.kneighbors(X_maj[border_majority])

        # Step 6 - informative minority indexes X_min
        informative_minority = np.unique(ind3.flatten())

        def closeness_factor(y, x, cf_th=self.cf_th, cmax=self.cmax):
            """
            Closeness factor according to the Eq (6)

            Args:
                y (np.array): training instance (border_majority)
                x (np.array): training instance (informative_minority)
                cf_th (float): cutoff threshold
                cmax (float): maximum values

            Returns:
                float: closeness factor
            """
            d = np.linalg.norm(y - x)/len(y)
            if d == 0.0:
                d = 0.1
            if 1.0/d < cf_th:
                f = 1.0/d
            else:
                f = cf_th
            return f/cf_th*cmax

        # Steps 7 - 9
        _logger.info(self.__class__.__name__ + ": " +
                     'computing closeness factors')
        closeness_factors = np.zeros(
            shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            bm_i = border_majority[i]
            for j in range(len(informative_minority)):
                im_j = informative_minority[j]
                closeness_factors[i, j] = closeness_factor(X_maj[bm_i],
                                                           X_min[im_j])

        _logger.info(self.__class__.__name__ + ": " +
                     'computing information weights')
        information_weights = np.zeros(
            shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            norm_factor = np.sum(closeness_factors[i, :])
            for j in range(len(informative_minority)):
                cf_ij = closeness_factors[i, j]
                information_weights[i, j] = cf_ij**2/norm_factor

        selection_weights = np.sum(information_weights, axis=0)
        selection_probabilities = selection_weights/np.sum(selection_weights)

        # Step 10
        _logger.info(self.__class__.__name__ + ": " + 'do clustering')
        n_clusters = min([len(X_min), self.M])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X_min)
        imin_labels = kmeans.labels_[informative_minority]

        clusters = [np.where(imin_labels == i)[0]
                    for i in range(np.max(kmeans.labels_)+1)]

        # Step 11
        samples = []

        # Step 12
        for i in range(n_to_sample):
            random_index = self.random_state.choice(informative_minority,
                                                    p=selection_probabilities)
            cluster_label = kmeans.labels_[random_index]
            cluster = clusters[cluster_label]
            random_index_in_cluster = self.random_state.choice(cluster)
            X_random = X_min[random_index]
            X_random_cluster = X_min[random_index_in_cluster]
            samples.append(self.sample_between_points(X_random,
                                                      X_random_cluster))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k1': self.k1,
                'k2': self.k2,
                'k3': self.k3,
                'M': self.M,
                'cf_th': self.cf_th,
                'cmax': self.cmax,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class PDFOS(OverSampling):
    """
    References:
        * BibTex::

            @article{pdfos,
                    title = "PDFOS: PDF estimation based over-sampling for
                                imbalanced two-class problems",
                    journal = "Neurocomputing",
                    volume = "138",
                    pages = "248 - 259",
                    year = "2014",
                    issn = "0925-2312",
                    doi = "https://doi.org/10.1016/j.neucom.2014.02.006",
                    author = "Ming Gao and Xia Hong and Sheng Chen and Chris
                                J. Harris and Emad Khalaf",
                    keywords = "Imbalanced classification, Probability density
                                function based over-sampling, Radial basis
                                function classifier, Orthogonal forward
                                selection, Particle swarm optimisation"
                    }

    Notes:
        * Not prepared for low-rank data.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_density_estimation]

    def __init__(self, proportion=1.0, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def _sample_by_kernel_density_estimation(self,
                                             X,
                                             n_to_sample,
                                             n_optimize=100):
        """
        Sample n_to_sample instances by kernel density estimation

        Args:
            X_min (np.array): minority data
            n_to_sample (int): number of instances to sample
            n_optimize (int): number of vectors used for the optimization
                                process
        """
        # dimensionality of the data
        m = len(X[0])

        # computing the covariance matrix of the data
        S = np.cov(X, rowvar=False)
        message = "Condition number of covariance matrix: %f"
        message = message % np.linalg.cond(S)
        _logger.info(self.__class__.__name__ + ": " + message)

        message = "Inputs size: %d" % len(X)
        _logger.info(self.__class__.__name__ + ": " + message)
        _logger.info(self.__class__.__name__ + ": " + "Input dim: %d" % m)

        S_mrank = np.linalg.matrix_rank(S, tol=1e-2)
        message = "Matrix rank of covariance matrix: %d" % S_mrank
        _logger.info(self.__class__.__name__ + ": " + message)

        # checking the rank of the matrix
        if S_mrank < m:
            message = "The covariance matrix is singular, fixing it by PCA"
            _logger.info(self.__class__.__name__ + ": " + message)
            message = "dim: %d, rank: %d, size: %d" % (m, S_mrank, len(X))
            _logger.info(self.__class__.__name__ + ": " + message)

            n_components = max([min([S_mrank, len(X)])-1, 2])
            if n_components == len(X[0]):
                return X.copy()

            pca = PCA(n_components=n_components)
            X_low_dim = pca.fit_transform(X)
            X_samp = self._sample_by_kernel_density_estimation(
                X_low_dim, n_to_sample, n_optimize)
            return pca.inverse_transform(X_samp)

        S_inv = np.linalg.inv(S)
        det = np.linalg.det(S)

        _logger.info(self.__class__.__name__ + ": " + "Determinant: %f" % det)

        def eq_9(i, j, sigma, X):
            """
            Eq (9) in the paper
            """
            tmp = np.dot(np.dot((X[j] - X[i]), S_inv), (X[j] - X[i]))
            numerator = (np.sqrt(2)*sigma)**(-m)*np.exp(-(1/(4*sigma**2))*tmp)
            denominator = ((2*np.pi)**(m/2))
            return numerator/denominator

        def eq_5(i, j, sigma, X):
            """
            Eq (5) in the paper
            """
            tmp = np.dot(np.dot((X[j] - X[i]), S_inv), (X[j] - X[i]))
            numerator = sigma**(-m)*np.exp(-(1/(2*sigma**2))*tmp)
            denominator = ((2.0*np.pi)**(m/2))
            return numerator/denominator

        def eq_5_0(sigma, X):
            """
            Eq (5) with the same vectors feeded in
            """
            return sigma**(-m)/((2.0*np.pi)**(m/2))

        def eq_8(i, j, sigma, X):
            """
            Eq (8) in the paper
            """
            e9 = eq_9(i, j, sigma, X)
            e5 = eq_5(i, j, sigma, X)
            return e9 - 2*e5

        def M(sigma, X):
            """
            Eq (7) in the paper
            """
            total = 0.0
            for i in range(len(X)):
                for j in range(len(X)):
                    total = total + eq_8(i, j, sigma, X)

            a = total/len(X)**2
            b = 2.0*eq_5_0(sigma, X)/len(X)
            return a + b

        # finding the best sigma parameter
        best_sigma = 0
        error = np.inf
        # the dataset is reduced to make the optimization more efficient
        domain = range(len(X))
        n_to_choose = min([len(X), n_optimize])
        X_reduced = X[self.random_state.choice(domain,
                                               n_to_choose,
                                               replace=False)]

        # we suppose that the data is normalized, thus, this search space
        # should be meaningful
        for sigma in np.logspace(-5, 2, num=20):
            e = M(sigma, X_reduced)
            if e < error:
                error = e
                best_sigma = sigma
        _logger.info(self.__class__.__name__ + ": " +
                     "best sigma found: %f" % best_sigma)

        # generating samples according to the
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X))
            samples.append(self.random_state.multivariate_normal(
                X[idx], best_sigma*S))

        return np.vstack(samples)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # scaling the data to aid numerical stability
        ss = StandardScaler()
        X_ss = ss.fit_transform(X)

        X_min = X_ss[y == self.min_label]

        # generating samples by kernel density estimation
        samples = self._sample_by_kernel_density_estimation(X_min,
                                                            n_to_sample,
                                                            n_optimize=100)

        return (np.vstack([X, ss.inverse_transform(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class IPADE_ID(OverSampling):
    """
    References:
        * BibTex::

            @article{ipade_id,
                    title = "Addressing imbalanced classification with
                                instance generation techniques: IPADE-ID",
                    journal = "Neurocomputing",
                    volume = "126",
                    pages = "15 - 28",
                    year = "2014",
                    note = "Recent trends in Intelligent Data Analysis Online
                                Data Processing",
                    issn = "0925-2312",
                    doi = "https://doi.org/10.1016/j.neucom.2013.01.050",
                    author = "Victoria López and Isaac Triguero and Cristóbal
                                J. Carmona and Salvador García and
                                Francisco Herrera",
                    keywords = "Differential evolution, Instance generation,
                                Nearest neighbor, Decision tree, Imbalanced
                                datasets"
                    }

    Notes:
        * According to the algorithm, if the addition of a majority sample
            doesn't improve the AUC during the DE optimization process,
            the addition of no further majority points is tried.
        * In the differential evolution the multiplication by a random number
            seems have a deteriorating effect, new scaling parameter added to
            fix this.
        * It is not specified how to do the evaluation.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_memetic,
                  OverSampling.cat_uses_classifier]

    def __init__(self,
                 F=0.1,
                 G=0.1,
                 OT=20,
                 max_it=40,
                 dt_classifier=DecisionTreeClassifier(random_state=2),
                 base_classifier=DecisionTreeClassifier(random_state=2),
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            F (float): control parameter of differential evolution
            G (float): control parameter of the evolution
            OT (int): number of optimizations
            max_it (int): maximum number of iterations for DE_optimization
            dt_classifier (obj): decision tree classifier object
            base_classifier (obj): classifier object
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater(F, 'F', 0)
        self.check_greater(G, 'G', 0)
        self.check_greater(OT, 'OT', 0)
        self.check_greater(max_it, 'max_it', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.F = F
        self.G = G
        self.OT = OT
        self.max_it = max_it
        self.dt_classifier = dt_classifier
        self.base_classifier = base_classifier
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        # as the OT and max_it parameters control the discovery of the feature
        # space it is enough to try sufficiently large numbers
        dt_classifiers = [DecisionTreeClassifier(random_state=2)]
        base_classifiers = [DecisionTreeClassifier(random_state=2)]
        parameter_combinations = {'F': [0.1, 0.2],
                                  'G': [0.1, 0.2],
                                  'OT': [30],
                                  'max_it': [40],
                                  'dt_classifier': dt_classifiers,
                                  'base_classifier': base_classifiers}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        mms = MinMaxScaler()
        X = mms.fit_transform(X)

        min_indices = np.where(y == self.min_label)[0]
        maj_indices = np.where(y == self.maj_label)[0]

        def DE_optimization(GS,
                            GS_y,
                            X,
                            y,
                            min_indices,
                            maj_indices,
                            classifier,
                            for_validation):
            """
            Implements the DE_optimization method of the paper.

            Args:
                GS (np.matrix): actual best training set
                GS_y (np.array): corresponding class labels
                X (np.matrix): complete training set
                y (np.array): all class labels
                min_indices (np.array): array of minority class labels in y
                maj_indices (np.array): array of majority class labels in y
                classifier (object): base classifier
                for_validation (np.array): array of indices for X used for
                                            validation

            Returns:
                np.matrix: optimized training set
            """
            # evaluate training set
            AUC_GS = evaluate_ID(
                GS, GS_y, X[for_validation], y[for_validation], classifier)

            # optimizing the training set
            for _ in range(self.max_it):
                GS_hat = []
                # doing the differential evolution
                for i in range(len(GS)):
                    if GS_y[i] == self.min_label:
                        r1, r2, r3 = self.random_state.choice(min_indices,
                                                              3,
                                                              replace=False)
                    else:
                        r1, r2, r3 = self.random_state.choice(maj_indices,
                                                              3,
                                                              replace=False)

                    random_value = self.random_state.random_sample()
                    force_G = X[r1] - X[i]
                    force_F = X[r2] - X[r3]
                    value = GS[i] + self.G*random_value * \
                        force_G + self.F*force_F
                    GS_hat.append(np.clip(value, 0.0, 1.0))

                # evaluating the current setting
                AUC_GS_hat = evaluate_ID(GS_hat,
                                         GS_y,
                                         X[for_validation],
                                         y[for_validation],
                                         classifier)

                if AUC_GS_hat > AUC_GS:
                    GS = GS_hat
                    AUC_GS = AUC_GS_hat

            return GS

        def evaluate_ID(GS, GS_y, TR, TR_y, base_classifier):
            """
            Implements the evaluate_ID function of the paper.

            Args:
                GS (np.matrix): actual training set
                GS_y (np.array): list of corresponding class labels
                TR (np.matrix): complete training set
                TR_y (np.array): all class labels
                base_classifier (object): classifier to be used

            Returns:
                float: ROC AUC score
            """
            base_classifier.fit(GS, GS_y)
            pred = base_classifier.predict_proba(TR)[:, np.where(
                base_classifier.classes_ == self.min_label)[0][0]]
            if len(np.unique(TR_y)) != 2:
                return 0.0
            return roc_auc_score(TR_y, pred)

        def evaluate_class(GS, GS_y, TR, TR_y, base_classifier):
            """
            Implements the evaluate_ID function of the paper.

            Args:
                GS (np.matrix): actual training set
                GS_y (np.array): list of corresponding class labels
                TR (np.matrix): complete training set
                TR_y (np.array): all class labels
                base_classifier (object): classifier to be used

            Returns:
                float: accuracy score
            """
            base_classifier.fit(GS, GS_y)
            pred = base_classifier.predict(TR)
            return accuracy_score(TR_y, pred)

        # Phase 1: Initialization
        _logger.info(self.__class__.__name__ + ": " + "Initialization")
        self.dt_classifier.fit(X, y)
        leafs = self.dt_classifier.apply(X)
        unique_leafs = np.unique(leafs)
        used_in_GS = np.repeat(False, len(X))
        for_validation = np.where(np.logical_not(used_in_GS))[0]

        # extracting mean elements of the leafs
        GS = []
        GS_y = []
        for u in unique_leafs:
            indices = np.where(leafs == u)[0]
            GS.append(np.mean(X[indices], axis=0))
            GS_y.append(mode(y[indices]))
            if len(indices) == 1:
                used_in_GS[indices[0]] = True

        # updating the indices of the validation set excluding those used in GS
        for_validation = np.where(np.logical_not(used_in_GS))[0]
        _logger.info(self.__class__.__name__ + ": " +
                     "Size of validation set %d" % len(for_validation))
        if len(np.unique(y[for_validation])) == 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "No minority samples in validation set")
            return X.copy(), y.copy()
        if len(np.unique(GS_y)) == 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "No minority samples in reduced dataset")
            return X.copy(), y.copy()

        # DE optimization takes place
        _logger.info(self.__class__.__name__ + ": " + "DE optimization")
        base_classifier = self.base_classifier.__class__(
            **(self.base_classifier.get_params()))
        GS = DE_optimization(GS, GS_y, X, y, min_indices,
                             maj_indices, base_classifier, for_validation)
        # evaluate results
        base_classifier = self.base_classifier.__class__(
            **(self.base_classifier.get_params()))
        AUC = evaluate_ID(GS, GS_y, X[for_validation],
                          y[for_validation], base_classifier)

        # Phase 2: Addition of new instances
        register_class = {self.min_label: 'optimizable',
                          self.maj_label: 'optimizable'}
        number_of_optimizations = {self.min_label: 0,
                                   self.maj_label: 0}
        accuracy_class = {self.min_label: 0, self.maj_label: 0}

        _logger.info(self.__class__.__name__ + ": " + "Starting optimization")
        while (AUC < 1.0
                and (register_class[self.min_label] == 'optimizable'
                     or register_class[self.maj_label] == 'optimizable')):
            less_accuracy = np.inf
            # loop in line 8
            for i in [self.min_label, self.maj_label]:
                # condition in line 9
                if register_class[i] == 'optimizable':
                    y_mask = y[for_validation] == i
                    class_for_validation = for_validation[y_mask]
                    bp = self.base_classifier.get_params()
                    base_classifier = self.base_classifier.__class__(**(bp))
                    accuracy_class[i] = evaluate_class(GS,
                                                       GS_y,
                                                       X[class_for_validation],
                                                       y[class_for_validation],
                                                       base_classifier)
                    if accuracy_class[i] < less_accuracy:
                        less_accuracy = accuracy_class[i]
                        target_class = i
            # conditional in line 17
            if (target_class == self.min_label
                    and number_of_optimizations[target_class] > 0):
                # it is not clear where does GS_trial coming from in line 18
                GS = DE_optimization(GS,
                                     GS_y,
                                     X,
                                     y,
                                     min_indices,
                                     maj_indices,
                                     base_classifier,
                                     for_validation)
            else:
                if target_class == self.min_label:
                    idx = self.random_state.choice(min_indices)
                else:
                    idx = self.random_state.choice(maj_indices)

                GS_trial = np.vstack([GS, X[idx]])
                GS_trial_y = np.hstack([GS_y, y[idx]])
                # removing idx from the validation set in order to keep
                # the validation fair
                for_validation_trial = for_validation.tolist()
                if idx in for_validation:
                    for_validation_trial.remove(idx)

                for_validation_trial = np.array(
                    for_validation_trial).astype(int)
                # doing optimization
                GS_trial = DE_optimization(GS_trial,
                                           GS_trial_y,
                                           X,
                                           y,
                                           min_indices,
                                           maj_indices,
                                           base_classifier,
                                           for_validation)

            # line 23
            bp = self.base_classifier.get_params()
            base_classifier = self.base_classifier.__class__(**(bp))

            AUC_trial = evaluate_ID(GS_trial,
                                    GS_trial_y,
                                    X[for_validation],
                                    y[for_validation],
                                    base_classifier)
            # conditional in line 24
            if AUC_trial > AUC:
                AUC = AUC_trial
                GS = GS_trial
                GS_y = GS_trial_y
                for_validation = for_validation_trial

                _logger.info(self.__class__.__name__ + ": " +
                             "Size of validation set %d" % len(for_validation))
                if len(np.unique(y[for_validation])) == 1:
                    _logger.info(self.__class__.__name__ + ": " +
                                 "No minority samples in validation set")
                    return X.copy(), y.copy()
                if len(np.unique(GS_y)) == 1:
                    _logger.info(self.__class__.__name__ + ": " +
                                 "No minority samples in reduced dataset")
                    return X.copy(), y.copy()

                number_of_optimizations[target_class] = 0
            else:
                # conditional in line 29
                if (target_class == self.min_label
                        and number_of_optimizations[target_class] < self.OT):
                    number_of_optimizations[target_class] += 1
                else:
                    register_class[target_class] = 'non-optimizable'

        return mms.inverse_transform(GS), GS_y

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'F': self.F,
                'G': self.G,
                'OT': self.OT,
                'max_it': self.max_it,
                'n_jobs': self.n_jobs,
                'dt_classifier': self.dt_classifier,
                'base_classifier': self.base_classifier,
                'random_state': self._random_state_init}


class RWO_sampling(OverSampling):
    """
    References:
        * BibTex::

            @article{rwo_sampling,
                    author = {Zhang, Huaxzhang and Li, Mingfang},
                    year = {2014},
                    month = {11},
                    pages = {},
                    title = {RWO-Sampling: A Random Walk Over-Sampling Approach
                                to Imbalanced Data Classification},
                    volume = {20},
                    booktitle = {Information Fusion}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self, proportion=1.0, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        stds = np.diag(np.std(X_min, axis=0)/np.sqrt(len(X_min)))

        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X_min))
            samples.append(self.random_state.multivariate_normal(X_min[idx],
                                                                 stds))

        return (np.vstack([X, samples]),
                np.hstack([y, np.array([self.min_label]*len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class NEATER(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{neater,
                            author={Almogahed, B. A. and Kakadiaris, I. A.},
                            booktitle={2014 22nd International Conference on
                                         Pattern Recognition},
                            title={NEATER: Filtering of Over-sampled Data
                                    Using Non-cooperative Game Theory},
                            year={2014},
                            volume={},
                            number={},
                            pages={1371-1376},
                            keywords={data handling;game theory;information
                                        filtering;NEATER;imbalanced data
                                        problem;synthetic data;filtering of
                                        over-sampled data using non-cooperative
                                        game theory;Games;Game theory;Vectors;
                                        Sociology;Statistics;Silicon;
                                        Mathematical model},
                            doi={10.1109/ICPR.2014.245},
                            ISSN={1051-4651},
                            month={Aug}}

    Notes:
        * Evolving both majority and minority probabilities as nothing ensures
            that the probabilities remain in the range [0,1], and they need to
            be normalized.
        * The inversely weighted function needs to be cut at some value (like
            the alpha level), otherwise it will overemphasize the utility of
            having differing neighbors next to each other.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 smote_n_neighbors=5,
                 b=5,
                 alpha=0.1,
                 h=20,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            smote_n_neighbors (int): number of neighbors in SMOTE sampling
            b (int): number of neighbors
            alpha (float): smoothing term
            h (int): number of iterations in evolution
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(smote_n_neighbors, "smote_n_neighbors", 1)
        self.check_greater_or_equal(b, "b", 1)
        self.check_greater_or_equal(alpha, "alpha", 0)
        self.check_greater_or_equal(h, "h", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.smote_n_neighbors = smote_n_neighbors
        self.b = b
        self.alpha = alpha
        self.h = h
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'smote_n_neighbors': [3, 5, 7],
                                  'b': [3, 5, 7],
                                  'alpha': [0.1],
                                  'h': [20]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # Applying SMOTE and ADASYN
        X_0, y_0 = SMOTE(proportion=self.proportion,
                         n_neighbors=self.smote_n_neighbors,
                         n_jobs=self.n_jobs,
                         random_state=self.random_state).sample(X, y)

        X_1, y_1 = ADASYN(n_neighbors=self.b,
                          n_jobs=self.n_jobs,
                          random_state=self.random_state).sample(X, y)

        X_new = np.vstack([X_0, X_1[len(X):]])
        y_new = np.hstack([y_0, y_1[len(y):]])

        X_syn = X_new[len(X):]

        if len(X_syn) == 0:
            return X.copy(), y.copy()

        X_all = X_new
        y_all = y_new

        # binary indicator indicating synthetic instances
        synthetic = np.hstack(
            [np.array([False]*len(X)), np.array([True]*len(X_syn))])

        # initializing strategy probabilities
        prob = np.zeros(shape=(len(X_all), 2))
        prob.fill(0.5)
        for i in range(len(X)):
            if y[i] == self.min_label:
                prob[i, 0], prob[i, 1] = 0.0, 1.0
            else:
                prob[i, 0], prob[i, 1] = 1.0, 0.0

        # Finding nearest neighbors, +1 as X_syn is part of X_all and nearest
        # neighbors will be themselves
        nn = NearestNeighbors(n_neighbors=self.b + 1, n_jobs=self.n_jobs)
        nn.fit(X_all)
        distances, indices = nn.kneighbors(X_syn)

        # computing distances
        dm = pairwise_distances(X_syn, X_all)
        dm[dm == 0] = 1e-8
        dm = 1.0/dm
        dm[dm > self.alpha] = self.alpha

        def wprob_mixed(prob, i):
            ind = indices[i][1:]
            term_0 = 1*prob[i][0]*prob[ind, 0]
            term_1 = dm[i, ind]*(prob[i][1]*prob[ind, 0] +
                                 prob[i][0]*prob[ind, 1])
            term_2 = 1*prob[i][1]*prob[ind, 1]
            return np.sum(term_0 + term_1 + term_2)

        def wprob_min(prob, i):
            term_0 = 0*prob[indices[i][1:], 0]
            term_1 = dm[i, indices[i][1:]]*(1*prob[indices[i][1:], 0] +
                                            0*prob[indices[i][1:], 1])
            term_2 = 1*prob[indices[i][1:], 1]
            return np.sum(term_0 + term_1 + term_2)

        def wprob_maj(prob, i):
            term_0 = 1*prob[indices[i][1:], 0]
            term_1 = dm[i, indices[i][1:]]*(0*prob[indices[i][1:], 0] +
                                            1*prob[indices[i][1:], 1])
            term_2 = 0*prob[indices[i][1:], 1]
            return np.sum(term_0 + term_1 + term_2)

        def utilities(prob):
            """
            Computes the utilit function

            Args:
                prob (np.matrix): strategy probabilities

            Returns:
                np.array, np.array, np.array: utility values, minority
                                                utilities, majority
                                                utilities
            """

            domain = range(len(X_syn))
            util_mixed = np.array([wprob_mixed(prob, i) for i in domain])
            util_mixed = np.hstack([np.array([0]*len(X)), util_mixed])

            util_min = np.array([wprob_min(prob, i) for i in domain])
            util_min = np.hstack([np.array([0]*len(X)), util_min])

            util_maj = np.array([wprob_maj(prob, i) for i in domain])
            util_maj = np.hstack([np.array([0]*len(X)), util_maj])

            return util_mixed, util_min, util_maj

        def evolution(prob, synthetic, alpha=self.alpha):
            """
            Executing one step of the probabilistic evolution

            Args:
                prob (np.matrix): strategy probabilities
                synthetic (np.array): flags of synthetic examples
                alpha (float): smoothing function

            Returns:
                np.matrix: updated probabilities
            """
            util_mixed, util_min, util_maj = utilities(prob)

            prob_new = prob.copy()
            synthetic_values = prob[:, 1] * \
                (alpha + util_min)/(alpha + util_mixed)
            prob_new[:, 1] = np.where(synthetic, synthetic_values, prob[:, 1])

            synthetic_values = prob[:, 0] * \
                (alpha + util_maj)/(alpha + util_mixed)
            prob_new[:, 0] = np.where(synthetic, synthetic_values, prob[:, 0])

            norm_factor = np.sum(prob_new, axis=1)

            prob_new[:, 0] = prob_new[:, 0]/norm_factor
            prob_new[:, 1] = prob_new[:, 1]/norm_factor

            return prob_new

        # executing the evolution
        for _ in range(self.h):
            prob = evolution(prob, synthetic)

        # determining final labels
        y_all[len(X):] = np.argmax(prob[len(X):], axis=1)

        return X_all, y_all

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'smote_n_neighbors': self.smote_n_neighbors,
                'b': self.b,
                'alpha': self.alpha,
                'h': self.h,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class DEAGO(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{deago,
                            author={Bellinger, C. and Japkowicz, N. and
                                        Drummond, C.},
                            booktitle={2015 IEEE 14th International
                                        Conference on Machine Learning
                                        and Applications (ICMLA)},
                            title={Synthetic Oversampling for Advanced
                                        Radioactive Threat Detection},
                            year={2015},
                            volume={},
                            number={},
                            pages={948-953},
                            keywords={radioactive waste;advanced radioactive
                                        threat detection;gamma-ray spectral
                                        classification;industrial nuclear
                                        facilities;Health Canadas national
                                        monitoring networks;Vancouver 2010;
                                        Isotopes;Training;Monitoring;
                                        Gamma-rays;Machine learning algorithms;
                                        Security;Neural networks;machine
                                        learning;classification;class
                                        imbalance;synthetic oversampling;
                                        artificial neural networks;
                                        autoencoders;gamma-ray spectra},
                            doi={10.1109/ICMLA.2015.58},
                            ISSN={},
                            month={Dec}}

    Notes:
        * There is no hint on the activation functions and amounts of noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_density_estimation,
                  OverSampling.cat_application]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 e=100,
                 h=0.3,
                 sigma=0.1,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            e (int): number of epochs
            h (float): fraction of number of hidden units
            sigma (float): training noise
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater(e, "e", 1)
        self.check_greater(h, "h", 0)
        self.check_greater(sigma, "sigma", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.e = e
        self.h = h
        self.sigma = sigma
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'e': [40],
                                  'h': [0.1, 0.2, 0.3, 0.4, 0.5],
                                  'sigma': [0.05, 0.1, 0.2]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # ugly hack to get reproducible results from keras with
        # tensorflow backend
        if isinstance(self._random_state_init, int):
            import os
            os.environ['PYTHONHASHSEED'] = str(self._random_state_init)
            import keras as K
            np.random.seed(self._random_state_init)
            import random
            random.seed(self._random_state_init)
            # from tensorflow import set_random_seed
            import tensorflow
            try:
                tensorflow.set_random_seed(self._random_state_init)
            except Exception as e:
                tensorflow.random.set_seed(self._random_state_init)
        else:
            seed = 127
            import os
            os.environ['PYTHONHASHSEED'] = str(seed)
            import keras as K
            np.random.seed(seed)
            import random
            random.seed(seed)
            # from tensorflow import set_random_seed
            import tensorflow
            try:
                tensorflow.compat.v1.set_random_seed(seed)
            except Exception as e:
                tensorflow.random.set_seed(self._random_state_init)

        from keras import backend as K
        import tensorflow as tf
        try:
            session_conf = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.compat.v1.Session(
                graph=tf.compat.v1.get_default_graph(), config=session_conf)
            K.set_session(sess)
        except Exception as e:
            session_conf = tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
            sess = tf.compat.v1.Session(
                graph=tf.compat.v1.get_default_graph(), config=session_conf)
            tf.compat.v1.keras.backend.set_session(sess)

        if not hasattr(self, 'Input'):
            from keras.layers import Input, Dense, GaussianNoise
            from keras.models import Model
            from tensorflow.keras.callbacks import EarlyStopping

            self.Input = Input
            self.Dense = Dense
            self.GaussianNoise = GaussianNoise
            self.Model = Model
            self.EarlyStopping = EarlyStopping

        # sampling by smote
        X_samp, y_samp = SMOTE(proportion=self.proportion,
                               n_neighbors=self.n_neighbors,
                               n_jobs=self.n_jobs,
                               random_state=self.random_state).sample(X, y)

        # samples to map to the manifold extracted by the autoencoder
        X_init = X_samp[len(X):]

        if len(X_init) == 0:
            return X.copy(), y.copy()

        # normalizing
        X_min = X[y == self.min_label]
        ss = StandardScaler()
        X_min_normalized = ss.fit_transform(X_min)
        X_init_normalized = ss.transform(X_init)

        # extracting dimensions
        d = len(X[0])
        encoding_d = max([2, int(np.rint(d*self.h))])

        message = "Input dimension: %d, encoding dimension: %d"
        message = message % (d, encoding_d)
        _logger.info(self.__class__.__name__ + ": " + message
                     )

        # constructing the autoencoder
        callbacks = [self.EarlyStopping(monitor='val_loss', patience=2)]

        input_layer = self.Input(shape=(d,))
        noise = self.GaussianNoise(self.sigma)(input_layer)
        encoded = self.Dense(encoding_d, activation='relu')(noise)
        decoded = self.Dense(d, activation='linear')(encoded)

        dae = self.Model(input_layer, decoded)
        dae.compile(optimizer='adadelta', loss='mean_squared_error')
        actual_epochs = max([self.e, int(5000.0/len(X_min))])

        if len(X_min) > 10:
            val_perc = 0.2
            val_num = int(val_perc*len(X_min))
            X_min_train = X_min_normalized[:-val_num]
            X_min_val = X_min_normalized[-val_num:]

            dae.fit(X_min_train,
                    X_min_train,
                    epochs=actual_epochs,
                    validation_data=(X_min_val, X_min_val),
                    callbacks=callbacks,
                    verbose=0)
        else:
            dae.fit(X_min_normalized, X_min_normalized,
                    epochs=actual_epochs, verbose=0)

        # mapping the initial samples to the manifold
        samples = ss.inverse_transform(dae.predict(X_init_normalized))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'e': self.e,
                'h': self.h,
                'sigma': self.sigma,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Gazzah(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{gazzah,
                            author={Gazzah, S. and Hechkel, A. and Essoukri
                                        Ben Amara, N. },
                            booktitle={2015 IEEE 12th International
                                        Multi-Conference on Systems,
                                        Signals Devices (SSD15)},
                            title={A hybrid sampling method for
                                    imbalanced data},
                            year={2015},
                            volume={},
                            number={},
                            pages={1-6},
                            keywords={computer vision;image classification;
                                        learning (artificial intelligence);
                                        sampling methods;hybrid sampling
                                        method;imbalanced data;
                                        diversification;computer vision
                                        domain;classical machine learning
                                        systems;intraclass variations;
                                        system performances;classification
                                        accuracy;imbalanced training data;
                                        training data set;over-sampling;
                                        minority class;SMOTE star topology;
                                        feature vector deletion;intra-class
                                        variations;distribution criterion;
                                        biometric data;true positive rate;
                                        Training data;Principal component
                                        analysis;Databases;Support vector
                                        machines;Training;Feature extraction;
                                        Correlation;Imbalanced data sets;
                                        Intra-class variations;Data analysis;
                                        Principal component analysis;
                                        One-against-all SVM},
                            doi={10.1109/SSD.2015.7348093},
                            ISSN={},
                            month={March}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_components=2,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_components (int): number of components in PCA analysis
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_components, "n_components", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_components': [2, 3, 4, 5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # do the oversampling
        pf_smote = polynom_fit_SMOTE(proportion=self.proportion,
                                     random_state=self.random_state)
        X_samp, y_samp = pf_smote.sample(X, y)
        X_min_samp = X_samp[len(X):]

        if len(X_min_samp) == 0:
            return X.copy(), y.copy()

        # do the undersampling
        X_maj = X[y == self.maj_label]

        # fitting the PCA model
        pca = PCA(n_components=min([len(X[0]), self.n_components]))
        X_maj_trans = pca.fit_transform(X_maj)
        R = np.sqrt(np.sum(np.var(X_maj_trans, axis=0)))
        # determining the majority samples to remove
        to_remove = np.where([np.linalg.norm(x) > R for x in X_maj_trans])[0]
        _logger.info(self.__class__.__name__ + ": " +
                     "Removing %d majority samples" % len(to_remove))
        # removing the majority samples
        X_maj = np.delete(X_maj, to_remove, axis=0)

        if len(X_min_samp) == 0:
            _logger.info("no samples added")
            return X.copy(), y.copy()

        return (np.vstack([X_maj, X_min_samp]),
                np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min_samp))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_components': self.n_components,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MCT(OverSampling):
    """
    References:
        * BibTex::

            @article{mct,
                    author = {Jiang, Liangxiao and Qiu, Chen and Li, Chaoqun},
                    year = {2015},
                    month = {03},
                    pages = {1551004},
                    title = {A Novel Minority Cloning Technique for
                                Cost-Sensitive Learning},
                    volume = {29},
                    booktitle = {International Journal of Pattern Recognition
                                    and Artificial Intelligence}
                    }

    Notes:
        * Mode is changed to median, distance is changed to Euclidean to
                support continuous features, and normalized.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_copy]

    def __init__(self, proportion=1.0, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # having continuous variables, the mode is replaced by median
        x_med = np.median(X_min, axis=0)
        distances = np.array([np.linalg.norm(x_med - x) for x in X_min])
        sums = np.sum(distances)
        if sums != 0:
            distances = distances/sums

        # distribution of copies is determined (Euclidean distance is a
        # dissimilarity measure which is changed to similarity by subtracting
        # from 1.0)
        distribution = (1.0 - distances)/(np.sum(1.0 - distances))

        if any(np.isnan(distribution)):
            _logger.warning(self.__class__.__name__ + ": " +
                            "NaN in the probability distribution")
            return X.copy(), y.copy()

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            samples.append(X_min[self.random_state.choice(
                np.arange(len(X_min)), p=distribution)])

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ADG(OverSampling):
    """
    References:
        * BibTex::

            @article{adg,
                    author = {Pourhabib, A. and Mallick, Bani K. and Ding, Yu},
                    year = {2015},
                    month = {16},
                    pages = {2695--2724},
                    title = {A Novel Minority Cloning Technique for
                                Cost-Sensitive Learning},
                    volume = {16},
                    journal = {Journal of Machine Learning Research}
                    }

    Notes:
        * This method has a lot of parameters, it becomes fairly hard to
            cross-validate thoroughly.
        * Fails if matrix is singular when computing alpha_star, fixed
            by PCA.
        * Singularity might be caused by repeating samples.
        * Maintaining the kernel matrix becomes unfeasible above a couple
            of thousand vectors.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 kernel='inner',
                 lam=1.0,
                 mu=1.0,
                 k=12,
                 gamma=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            kernel (str): 'inner'/'rbf_x', where x is a float, the bandwidth
            lam (float): lambda parameter of the method
            mu (float): mu parameter of the method
            k (int): number of samples to generate in each iteration
            gamma (float): gamma parameter of the method
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)

        if kernel != 'inner' and not kernel.startswith('rbf'):
            raise ValueError(self.__class__.__name__ + ": " +
                             'Kernel function %s not supported' % kernel)
        elif kernel.startswith('rbf'):
            par = float(kernel.split('_')[-1])
            if par <= 0.0:
                raise ValueError(self.__class__.__name__ + ": " +
                                 'Kernel parameter %f is not supported' % par)

        self.check_greater(lam, 'lam', 0)
        self.check_greater(mu, 'mu', 0)
        self.check_greater_or_equal(k, 'k', 1)
        self.check_greater(gamma, 'gamma', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.kernel = kernel
        self.lam = lam
        self.mu = mu
        self.k = k
        self.gamma = gamma
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'kernel': ['inner', 'rbf_0.5',
                                             'rbf_1.0', 'rbf_2.0'],
                                  'lam': [1.0, 2.0],
                                  'mu': [1.0, 2.0],
                                  'k': [12],
                                  'gamma': [1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        def bic_score(kmeans, X):
            """
            Compute BIC score for clustering

            Args:
                kmeans (sklearn.KMeans): kmeans object
                X (np.matrix):  clustered data

            Returns:
                float: bic value

            Inspired by https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
            """  # noqa
            # extract descriptors of the clustering
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_
            n_clusters = kmeans.n_clusters
            n_in_clusters = np.bincount(cluster_labels)
            N, d = X.shape

            # compute variance for all clusters beforehand

            def sum_norm_2(i):
                return np.sum(np.linalg.norm(X[cluster_labels == i] -
                                             cluster_centers[i])**2)

            cluster_variances = [sum_norm_2(i) for i in range(n_clusters)]
            term_0 = (1.0)/((N - n_clusters) * d)
            term_1 = np.sum(cluster_variances)
            clustering_variance = term_0 * term_1

            const_term = 0.5 * n_clusters * np.log(N) * (d+1)

            def bic_comp(i):
                term_0 = n_in_clusters[i] * np.log(n_in_clusters[i])
                term_1 = n_in_clusters[i] * np.log(N)
                term_2 = (((n_in_clusters[i] * d) / 2)
                          * np.log(2*np.pi*clustering_variance))
                term_3 = ((n_in_clusters[i] - 1) * d / 2)

                return term_0 - term_1 - term_2 - term_3

            bic = np.sum([bic_comp(i) for i in range(n_clusters)]) - const_term

            return bic

        def xmeans(X, r=(1, 10)):
            """
            Clustering with BIC based n_cluster selection

            Args:
                X (np.matrix): data to cluster
                r (tuple): lower and upper bound on the number of clusters

            Returns:
                sklearn.KMeans: clustering with lowest BIC score
            """
            best_bic = np.inf
            best_clustering = None

            # do clustering for all n_clusters in the specified range
            for k in range(r[0], min([r[1], len(X)])):
                kmeans = KMeans(n_clusters=k,
                                random_state=self.random_state).fit(X)

                bic = bic_score(kmeans, X)
                if bic < best_bic:
                    best_bic = bic
                    best_clustering = kmeans

            return best_clustering

        def xgmeans(X, r=(1, 10)):
            """
            Gaussian mixture with BIC to select the optimal number
            of components

            Args:
                X (np.matrix): data to cluster
                r (tuple): lower and upper bound on the number of components

            Returns:
                sklearn.GaussianMixture: Gaussian mixture model with the
                                            lowest BIC score
            """
            best_bic = np.inf
            best_mixture = None

            # do model fitting for all n_components in the specified range
            for k in range(r[0], min([r[1], len(X)])):
                gmm = GaussianMixture(
                    n_components=k, random_state=self.random_state).fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_mixture = gmm

            return best_mixture

        def evaluate_matrices(X, y, kernel=np.inner):
            """
            The function evaluates the matrices specified in the method.

            Args:
                X (np.matrix): features
                y (np.array): target labels
                kernel (function): the kernel function to be used

            Returns:
                np.matrix, np.matrix, int, int, np.matrix, np.array,
                np.matrix, np.matrix, np.matrix
                np.array, np.matrix, np.matrix, np.matrix, np.matrix:
                    X_minux, X_plus, l_minus, l_plus, X, y, K, M_plus, M_minus,
                    M, K_plus, K_minus, N_plus, n_minus using the notations of
                    the paper, X and y are ordered by target labels
            """
            X_minus = X[y == self.maj_label]
            X_plus = X[y == self.min_label]
            l_minus = len(X_minus)
            l_plus = len(X_plus)

            X = np.vstack([X_minus, X_plus])
            y = np.hstack([np.array([self.maj_label]*l_minus),
                           np.array([self.min_label]*l_plus)])

            K = pairwise_distances(X, X, metric=kernel)
            M_plus = np.mean(K[:, len(X_minus):], axis=1)
            M_minus = np.mean(K[:, :len(X_minus)], axis=1)
            M = np.dot(M_minus - M_plus, M_minus - M_plus)

            K_minus = K[:, :len(X_minus)]
            K_plus = K[:, len(X_minus):]

            return (X_minus, X_plus, l_minus, l_plus, X, y, K,
                    M_plus, M_minus, M, K_plus, K_minus)

        # Implementation of the technique, following the steps and notations
        # of the paper
        q = n_to_sample

        # instantiating the proper kernel function, the parameter of the RBF
        # is supposed to be the denominator in the Gaussian
        if self.kernel == 'inner':
            kernel_function = np.inner
        else:
            kf = self.kernel.split('_')
            if kf[0] == 'rbf':
                d = float(kf[1])
                def kernel_function(
                    x, y): return np.exp(-np.linalg.norm(x - y)**2/d)

        # Initial evaluation of the matrices
        (X_minus, X_plus, l_minus, l_plus, X, y, K, M_plus, M_minus,
         M, K_plus, K_minus) = evaluate_matrices(X,
                                                 y,
                                                 kernel=kernel_function)
        # The computing of N matrix is factored into two steps, computing
        # N_plus and N_minus this is used to improve efficiency
        K_plus2 = np.dot(K_plus, K_plus.T)
        K_plus_sum = np.sum(K_plus, axis=1)
        K_plus_diad = np.outer(K_plus_sum, K_plus_sum)/l_plus

        K_minus2 = np.dot(K_minus, K_minus.T)
        K_minus_sum = np.sum(K_minus, axis=1)
        K_minus_diad = np.outer(K_minus_sum, K_minus_sum)/l_minus

        N = K_plus2 - K_plus_diad + K_minus2 - K_minus_diad

        X_plus_hat = X_plus.copy()
        l_minus = len(X_minus)

        early_stop = False
        total_added = 0
        # executing the sample generation
        while q > 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "Starting iteration with q=%d" % q)
            # step 1
            clusters = xmeans(X_plus_hat)
            l_c = np.array([np.sum(clusters.labels_ == i)
                            for i in range(clusters.n_clusters)])

            # step 2
            k_c = ((1.0/l_c)/(np.sum(1.0/l_c))*self.k).astype(int)
            k_c[k_c == 0] = 1
            lam_c, mu_c = self.lam/l_c, self.mu/l_c

            # step 3
            omega = - np.sum([k_c[i]*(lam_c[i])**2/(4*mu_c[i]**2)
                              for i in range(len(k_c))])
            nu_c = - 0.5*k_c*lam_c
            M_plus_c = [np.mean(K[:, np.arange(len(X_minus), len(X))[
                clusters.labels_ == i]]) for i in range(len(k_c))]

            # step 4
            A = (M - self.gamma*N) - omega*K
            b = np.sum([(M_minus - M_plus_c[i])*nu_c[i]
                        for i in range(len(k_c))], axis=0)
            try:
                alpha_star = np.linalg.solve(A, b)
            except Exception as e:
                # handling the issue of singular matrix
                _logger.warning(self.__class__.__name__ +
                                ": " + "Singular matrix")
                # deleting huge data structures
                if q == n_to_sample:
                    if len(X[0]) == 1:
                        return None, None
                    K, K_plus, K_minus = None, None, None
                    n_components = int(np.sqrt(len(X[0])))
                    pca = PCA(n_components=n_components).fit(X)

                    message = "reducing dimensionality to %d" % n_components
                    _logger.warning(self.__class__.__name__ + ": " + message)
                    X_trans = pca.transform(X)
                    adg = ADG(proportion=self.proportion,
                              kernel=self.kernel,
                              lam=self.lam,
                              mu=self.mu,
                              k=self.k,
                              gamma=self.gamma,
                              random_state=self.random_state)
                    X_samp, y_samp = adg.sample(X_trans, y)
                    if X_samp is not None:
                        return pca.inverse_transform(X_samp), y_samp
                    else:
                        return X.copy(), y.copy()
                else:
                    q = int(q/2)
                continue

            # step 5
            mixture = xgmeans(X_plus)

            # step 6
            try:
                Z = mixture.sample(q)[0]
            except Exception as e:
                message = "sampling error in sklearn.mixture.GaussianMixture"
                _logger.warning(
                    self.__class__.__name__ + ": " + message)
                return X.copy(), y.copy()

            # step 7
            # computing the kernel matrix of generated samples with all samples
            K_10 = pairwise_distances(Z, X, metric=kernel_function)
            mask_inner_prod = np.where(np.inner(K_10, alpha_star) > 0)[0]
            Z_hat = Z[mask_inner_prod]

            if len(Z_hat) == 0:
                q = int(q/2)
                continue

            _logger.info(self.__class__.__name__ + ": " +
                         "number of vectors added: %d/%d" % (len(Z_hat), q))

            # step 8
            # this step is not used for anything, the identified clusters are
            # only used in step 13 of the paper, however, the values set
            # (M_plus^c) are overwritten in step 3 of the next iteration

            # step 9
            X_plus_hat = np.vstack([X_plus_hat, Z_hat])
            l_plus = len(X_plus_hat)

            # step 11 - 16
            # these steps have been reorganized a bit for efficient
            # calculations

            pairwd = pairwise_distances(Z_hat, Z_hat, metric=kernel_function)
            K = np.block([[K, K_10[mask_inner_prod].T],
                          [K_10[mask_inner_prod], pairwd]])

            K_minus = K[:, :l_minus]
            K_plus = K[:, l_minus:]

            # step 10
            X = np.vstack([X_minus, X_plus_hat])
            y = np.hstack([y, np.repeat(self.min_label, len(Z_hat))])

            if early_stop is True:
                break

            M_plus = np.mean(K_plus, axis=1)
            M_minus = np.mean(K_minus, axis=1)

            # step 13 is already involved in the core of the loop
            M = np.dot(M_minus - M_plus, M_minus - M_plus)

            l_new = len(Z_hat)
            total_added = total_added + l_new

            K_minus2_01 = np.dot(K_minus[:-l_new:], K_minus[-l_new:].T)
            K_minus2 = np.block([[K_minus2, K_minus2_01],
                                 [K_minus2_01.T, np.dot(K_minus[-l_new:],
                                                        K_minus[-l_new:].T)]])
            K_minus_sum = M_minus*len(K_minus)

            K_plus2 = K_plus2 + np.dot(K_plus[:-l_new, l_new:],
                                       K_plus[:-l_new, l_new:].T)

            K_plus2_01 = np.dot(K_plus[:-l_new], K_plus[-l_new:].T)

            K_plus2 = np.block([[K_plus2, K_plus2_01],
                                [K_plus2_01.T, np.dot(K_plus[-l_new:],
                                                      K_plus[-l_new:].T)]])

            K_plus_sum = M_plus*len(K_plus)

            N = K_plus2 - np.outer(K_plus_sum/l_plus, K_plus_sum) + \
                K_minus2 - np.outer(K_minus_sum/l_minus, K_minus_sum)

            # step 17
            if l_new/total_added < 0.01:
                early_stop = True
            else:
                q = int(q/2)

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'kernel': self.kernel,
                'lam': self.lam,
                'mu': self.mu,
                'k': self.k,
                'gamma': self.gamma,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_IPF(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_ipf,
                        title = "SMOTE–IPF: Addressing the noisy and borderline
                                    examples problem in imbalanced
                                    classification by a re-sampling method
                                    with filtering",
                        journal = "Information Sciences",
                        volume = "291",
                        pages = "184 - 203",
                        year = "2015",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2014.08.051",
                        author = "José A. Sáez and Julián Luengo and Jerzy
                                    Stefanowski and Francisco Herrera",
                        keywords = "Imbalanced classification,
                                        Borderline examples,
                                        Noisy data,
                                        Noise filters,
                                        SMOTE"
                        }
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_uses_classifier]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_folds=9,
                 k=3,
                 p=0.01,
                 voting='majority',
                 classifier=DecisionTreeClassifier(random_state=2),
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE sampling
            n_folds (int): the number of partitions
            k (int): used in stopping condition
            p (float): percentage value ([0,1]) used in stopping condition
            voting (str): 'majority'/'consensus'
            classifier (obj): classifier object
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_folds, "n_folds", 2)
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater_or_equal(p, "p", 0)
        self.check_isin(voting, "voting", ['majority', 'consensus'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_folds = n_folds
        self.k = k
        self.p = p
        self.voting = voting
        self.classifier = classifier
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        classifiers = [DecisionTreeClassifier(random_state=2)]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_folds': [9],
                                  'k': [3],
                                  'p': [0.01],
                                  'voting': ['majority', 'consensus'],
                                  'classifier': classifiers}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # do SMOTE sampling
        X_samp, y_samp = SMOTE(self.proportion,
                               self.n_neighbors,
                               n_jobs=self.n_jobs,
                               random_state=self.random_state).sample(X, y)

        n_folds = min([self.n_folds, np.sum(y == self.min_label)])

        condition = 0
        while True:
            # validating the sampled dataset
            validator = StratifiedKFold(n_folds)
            predictions = []
            for train_index, _ in validator.split(X_samp, y_samp):
                self.classifier.fit(X_samp[train_index], y_samp[train_index])
                predictions.append(self.classifier.predict(X_samp))

            # do decision based on one of the voting schemes
            if self.voting == 'majority':
                pred_votes = (np.mean(predictions, axis=0) > 0.5).astype(int)
                to_remove = np.where(np.not_equal(pred_votes, y_samp))[0]
            elif self.voting == 'consensus':
                pred_votes = (np.mean(predictions, axis=0) > 0.5).astype(int)
                sum_votes = np.sum(predictions, axis=0)
                to_remove = np.where(np.logical_and(np.not_equal(
                    pred_votes, y_samp), np.equal(sum_votes, self.n_folds)))[0]
            else:
                message = 'Voting scheme %s is not implemented' % self.voting
                raise ValueError(self.__class__.__name__ + ": " + message)

            # delete samples incorrectly classified
            _logger.info(self.__class__.__name__ + ": " +
                         'Removing %d elements' % len(to_remove))
            X_samp = np.delete(X_samp, to_remove, axis=0)
            y_samp = np.delete(y_samp, to_remove)

            # if the number of samples removed becomes small or k iterations
            # were done quit
            if len(to_remove) < len(X_samp)*self.p:
                condition = condition + 1
            else:
                condition = 0
            if condition >= self.k:
                break

        return X_samp, y_samp

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_folds': self.n_folds,
                'k': self.k,
                'p': self.p,
                'voting': self.voting,
                'n_jobs': self.n_jobs,
                'classifier': self.classifier,
                'random_state': self._random_state_init}


class KernelADASYN(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{kernel_adasyn,
                            author={Tang, B. and He, H.},
                            booktitle={2015 IEEE Congress on Evolutionary
                                        Computation (CEC)},
                            title={KernelADASYN: Kernel based adaptive
                                    synthetic data generation for
                                    imbalanced learning},
                            year={2015},
                            volume={},
                            number={},
                            pages={664-671},
                            keywords={learning (artificial intelligence);
                                        pattern classification;
                                        sampling methods;KernelADASYN;
                                        kernel based adaptive synthetic
                                        data generation;imbalanced
                                        learning;standard classification
                                        algorithms;data distribution;
                                        minority class decision rule;
                                        expensive minority class data
                                        misclassification;kernel based
                                        adaptive synthetic over-sampling
                                        approach;imbalanced data
                                        classification problems;kernel
                                        density estimation methods;Kernel;
                                        Estimation;Accuracy;Measurement;
                                        Standards;Training data;Sampling
                                        methods;Imbalanced learning;
                                        adaptive over-sampling;kernel
                                        density estimation;pattern
                                        recognition;medical and
                                        healthcare data learning},
                            doi={10.1109/CEC.2015.7256954},
                            ISSN={1089-778X},
                            month={May}}

    Notes:
        * The method of sampling was not specified, Markov Chain Monte Carlo
            has been implemented.
        * Not prepared for improperly conditioned covariance matrix.
    """

    categories = [OverSampling.cat_density_estimation,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 k=5,
                 h=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            k (int): number of neighbors in the nearest neighbors component
            h (float): kernel bandwidth
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, 'k', 1)
        self.check_greater(h, 'h', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k
        self.h = h
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k': [5, 7, 9],
                                  'h': [0.01, 0.02, 0.05, 0.1, 0.2,
                                        0.5, 1.0, 2.0, 10.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the nearest neighbors model
        nn = NearestNeighbors(n_neighbors=min([len(X_min), self.k+1]),
                              n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X_min)

        # computing majority score
        r = np.array([np.sum(y[indices[i][1:]] == self.maj_label)
                      for i in range(len(X_min))])

        if np.sum(r > 0) < 2:
            message = ("majority score is 0 for all or all but one "
                       "minority samples")
            _logger.info(self.__class__.__name__ + ": " + message)
            return X.copy(), y.copy()

        r = r/np.sum(r)

        # kernel density function
        def p_x(x):
            """
            Returns minority density value at x

            Args:
                x (np.array): feature vector

            Returns:
                float: density value
            """
            result = 1.0/(len(X_min)*self.h)
            result = result*(1.0/(np.sqrt(2*np.pi)*self.h)**len(X[0]))

            exp_term = np.exp(-0.5*np.linalg.norm(x - X_min, axis=1)**2/self.h)
            return result*np.inner(r, exp_term)

        samples = []
        it = 0

        # parameters of the Monte Carlo sampling
        burn_in = 1000
        periods = 50

        # covariance is used to generate a random sample in the neighborhood
        covariance = np.cov(X_min[r > 0], rowvar=False)

        if len(covariance) > 1 and np.linalg.cond(covariance) > 10000:
            message = ("reducing dimensions due to inproperly conditioned"
                       "covariance matrix")
            _logger.info(self.__class__.__name__ + ": " + message)

            if len(X[0]) <= 2:
                _logger.info(self.__class__.__name__ +
                             ": " + "matrix ill-conditioned")
                return X.copy(), y.copy()

            n_components = int(np.rint(len(covariance)/2))

            pca = PCA(n_components=n_components)
            X_trans = pca.fit_transform(X)

            ka = KernelADASYN(proportion=self.proportion,
                              k=self.k,
                              h=self.h,
                              random_state=self.random_state)

            X_samp, y_samp = ka.sample(X_trans, y)
            return pca.inverse_transform(X_samp), y_samp

        # starting Markov-Chain Monte Carlo for sampling
        x_old = X_min[self.random_state.choice(np.where(r > 0)[0])]
        p_old = p_x(x_old)

        # Cholesky decomposition
        L = np.linalg.cholesky(covariance)

        while len(samples) < n_to_sample:
            x_new = x_old + \
                np.dot(self.random_state.normal(size=len(x_old)), L)
            p_new = p_x(x_new)

            alpha = p_new/p_old
            u = self.random_state.random_sample()
            if u < alpha:
                x_old = x_new
                p_old = p_new
            else:
                pass

            it = it + 1
            if it % periods == 0 and it > burn_in:
                samples.append(x_old)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'h': self.h,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MOT2LD(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{mot2ld,
                            author="Xie, Zhipeng
                            and Jiang, Liyang
                            and Ye, Tengju
                            and Li, Xiaoli",
                            editor="Renz, Matthias
                            and Shahabi, Cyrus
                            and Zhou, Xiaofang
                            and Cheema, Muhammad Aamir",
                            title="A Synthetic Minority Oversampling Method
                                    Based on Local Densities in Low-Dimensional
                                    Space for Imbalanced Learning",
                            booktitle="Database Systems for Advanced
                                        Applications",
                            year="2015",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="3--18",
                            isbn="978-3-319-18123-3"
                            }

    Notes:
        * Clusters might contain 1 elements, and all points can be filtered
            as noise.
        * Clusters might contain 0 elements as well, if all points are filtered
            as noise.
        * The entire clustering can become empty.
        * TSNE is very slow when the number of instances is over a couple
            of 1000
    """

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_components=2,
                 k=5,
                 d_cut='auto',
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_components (int): number of components for stochastic
                                neighborhood embedding
            k (int): number of neighbors in the nearest neighbor component
            d_cut (float/str): distance cut value/'auto' for automated
                                selection
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_components, 'n_component', 1)
        self.check_greater_or_equal(k, 'k', 1)
        if isinstance(d_cut, float) or isinstance(d_cut, int):
            if d_cut <= 0:
                raise ValueError(self.__class__.__name__ +
                                 ": " + 'Non-positive d_cut is not allowed')
        elif d_cut != 'auto':
            raise ValueError(self.__class__.__name__ + ": " +
                             'd_cut value %s not implemented' % d_cut)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.k = k
        self.d_cut = d_cut
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_components': [2],
                                  'k': [3, 5, 7],
                                  'd_cut': ['auto']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        _logger.info(self.__class__.__name__ + ": " +
                     ("starting TSNE n: %d d: %d" % (len(X), len(X[0]))))
        # do the stochastic embedding
        X_tsne = TSNE(self.n_components,
                      random_state=self.random_state,
                      perplexity=10,
                      n_iter_without_progress=100,
                      n_iter=500,
                      verbose=3).fit_transform(X)
        X_min = X_tsne[y == self.min_label]
        _logger.info(self.__class__.__name__ + ": " + "TSNE finished")

        # fitting nearest neighbors model for all training data
        n_neighbors = min([len(X_min), self.k + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_tsne)
        distances, indices = nn.kneighbors(X_min)

        if isinstance(self.d_cut, float):
            d_cut = self.d_cut
        elif self.d_cut == 'auto':
            d_cut = np.max(distances[:, 1])

        # fitting nearest neighbors model to the minority data
        nn_min = NearestNeighbors(n_neighbors=len(X_min), n_jobs=self.n_jobs)
        nn_min.fit(X_min)
        distances_min, indices_min = nn_min.kneighbors(X_min)

        def n_rad_neighbors(x):
            x = x.reshape(1, -1)
            return len(nn.radius_neighbors(x, d_cut, return_distance=False)[0])

        # extracting the number of neighbors in a given radius
        rho = np.array([n_rad_neighbors(x) for x in X_min])
        closest_highest = []
        delta = []

        # implementation of the density peak clustering algorithm
        # based on http://science.sciencemag.org/content/344/6191/1492.full
        for i in range(len(rho)):
            closest_neighbors = indices_min[i]
            closest_densities = rho[closest_neighbors]
            closest_highs = np.where(closest_densities > rho[i])[0]

            if len(closest_highs) > 0:
                closest_highest.append(closest_highs[0])
                delta.append(distances_min[i][closest_highs[0]])
            else:
                closest_highest.append(-1)
                delta.append(np.max(distances_min))

        to_sort = zip(rho, delta, np.arange(len(rho)))
        r, d, idx = zip(*sorted(to_sort, key=lambda x: x[0]))
        r, d, idx = np.array(r), np.array(d), np.array(idx)

        if len(d) < 3:
            return X.copy(), y.copy()

        widths = np.arange(1, int(len(r)/2))
        peak_indices = np.array(ssignal.find_peaks_cwt(d, widths=widths))

        if len(peak_indices) == 0:
            _logger.info(self.__class__.__name__ + ": " + "no peaks found")
            return X.copy(), y.copy()

        cluster_center_indices = idx[peak_indices]
        cluster_centers = X_min[cluster_center_indices]

        # finding closest cluster center to minority points and deriving
        # cluster labels
        nn_cluster = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
        nn_cluster.fit(cluster_centers)
        dist_cluster, ind_cluster = nn_cluster.kneighbors(X_min)
        cluster_labels = ind_cluster[:, 0]

        # computing local minority counts and determining noisy samples
        def n_min_y(i):
            return np.sum(y[indices[i][1:]] == self.min_label)

        local_minority_count = np.array(
            [n_min_y(i) for i in range(len(X_min))])

        noise = np.where(np.logical_or(rho == 1, local_minority_count == 0))[0]

        # determining importance scores
        importance = local_minority_count/rho
        prob = importance
        prob[noise] = 0.0
        prob = prob/np.sum(prob)

        # extracting cluster indices
        cluster_indices = [np.where(cluster_labels == i)[0]
                           for i in range(np.max(cluster_labels) + 1)]
        # removing noise from clusters
        cluster_indices = [list(set(c).difference(set(noise)))
                           for c in cluster_indices]

        # checking if clustering is empty
        empty_clustering = True
        for i in range(len(cluster_indices)):
            if len(cluster_indices[i]) > 0:
                empty_clustering = False

        if empty_clustering:
            _logger.info(self.__class__.__name__ + ": " + "Empty clustering")
            return X.copy(), y.copy()

        cluster_sizes = np.array([len(c) for c in cluster_indices])
        cluster_indices_size_0 = np.where(cluster_sizes == 0)[0]
        for i in range(len(prob)):
            if cluster_labels[i] in cluster_indices_size_0:
                prob[i] = 0.0
        prob = prob/np.sum(prob)

        # carrying out the sampling
        X_min = X[y == self.min_label]
        samples = []
        while len(samples) < n_to_sample:
            # random sample according to the distribution computed
            random_idx = self.random_state.choice(np.arange(len(X_min)),
                                                  p=prob)

            # cluster label of the random minority sample
            cluster_label = cluster_labels[random_idx]
            if cluster_label == -1:
                continue

            if len(cluster_indices[cluster_label]) == 0:
                continue
            elif len(cluster_indices[cluster_label]) == 1:
                # if the cluster has only 1 elements, it is repeated
                samples.append(X_min[random_idx])
                continue
            else:
                # otherwise a random cluster index is selected for sample
                # generation
                clus = cluster_indices[cluster_label]
                random_neigh_in_clus_idx = self.random_state.choice(clus)
                while random_idx == random_neigh_in_clus_idx:
                    random_neigh_in_clus_idx = self.random_state.choice(clus)

                X_rand = X_min[random_idx]
                X_in_clus = X_min[random_neigh_in_clus_idx]
                samples.append(self.sample_between_points(X_rand, X_in_clus))

        return (np.vstack([np.delete(X, noise, axis=0), np.vstack(samples)]),
                np.hstack([np.delete(y, noise),
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_components': self.n_components,
                'k': self.k,
                'd_cut': self.d_cut,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class V_SYNTH(OverSampling):
    """
    References:
        * BibTex::

            @article{v_synth,
                     author = {Young,Ii, William A. and Nykl, Scott L. and
                                Weckman, Gary R. and Chelberg, David M.},
                     title = {Using Voronoi Diagrams to Improve
                                Classification Performances when Modeling
                                Imbalanced Datasets},
                     journal = {Neural Comput. Appl.},
                     issue_date = {July      2015},
                     volume = {26},
                     number = {5},
                     month = jul,
                     year = {2015},
                     issn = {0941-0643},
                     pages = {1041--1054},
                     numpages = {14},
                     url = {http://dx.doi.org/10.1007/s00521-014-1780-0},
                     doi = {10.1007/s00521-014-1780-0},
                     acmid = {2790665},
                     publisher = {Springer-Verlag},
                     address = {London, UK, UK},
                     keywords = {Data engineering, Data mining, Imbalanced
                                    datasets, Knowledge extraction,
                                    Numerical algorithms, Synthetic
                                    over-sampling},
                    }

    Notes:
        * The proposed encompassing bounding box generation is incorrect.
        * Voronoi diagram generation in high dimensional spaces is instable
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_components=3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_components (int): number of components for PCA
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_components, "n_component", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_components': [3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # creating the bounding box
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        mins = mins - 0.1*np.abs(mins)
        maxs = maxs + 0.1*np.abs(maxs)

        dim = len(X[0])

        def random_min_maxs():
            return np.where(self.random_state.randint(0, 1, size=dim) == 0,
                            mins,
                            maxs)

        n_bounding_box = min([100, len(X[0])])
        bounding_box = [random_min_maxs() for i in range(n_bounding_box)]
        X_bb = np.vstack([X, bounding_box])

        # applying PCA to reduce the dimensionality of the data
        n_components = min([len(X[0]), self.n_components])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_bb)
        y_pca = np.hstack([y, np.repeat(-1, len(bounding_box))])

        dm = pairwise_distances(X_pca)
        to_remove = []
        for i in range(len(dm)):
            for j in range(i+1, len(dm)):
                if dm[i, j] < 0.001:
                    to_remove.append(i)
        X_pca = np.delete(X_pca, to_remove, axis=0)
        y_pca = np.delete(y_pca, to_remove)

        # doing the Voronoi tessellation
        voronoi = sspatial.Voronoi(X_pca)

        # extracting those ridge point pairs which are candidates for
        # generating an edge between two cells of different class labels
        candidate_face_generators = []
        for i, r in enumerate(voronoi.ridge_points):
            if r[0] < len(y) and r[1] < len(y) and not y[r[0]] == y[r[1]]:
                candidate_face_generators.append(i)

        if len(candidate_face_generators) == 0:
            return X.copy(), y.copy()

        # generating samples
        samples = []
        for _ in range(n_to_sample):
            # randomly choosing a pair from the ridge point pairs of different
            # labels
            random_face = self.random_state.choice(candidate_face_generators)

            # extracting the vertices of the face between the points
            ridge_vertices = voronoi.ridge_vertices[random_face]
            face_vertices = voronoi.vertices[ridge_vertices]

            # creating a random vector for sampling the face (supposed to be
            # convex)
            w = self.random_state.random_sample(size=len(X_pca[0]))
            w = w/np.sum(w)

            # initiating a sample point on the face
            sample_point_on_face = np.zeros(len(X_pca[0]))
            for i in range(len(X_pca[0])):
                sample_point_on_face += w[i]*face_vertices[i]

            # finding the ridge point with the minority label
            if y[voronoi.ridge_points[random_face][0]] == self.min_label:
                h = voronoi.points[voronoi.ridge_points[random_face][0]]
            else:
                h = voronoi.points[voronoi.ridge_points[random_face][1]]

            # generating a point between the minority ridge point and the
            # random point on the face
            samples.append(self.sample_between_points(sample_point_on_face,
                                                      h))

        return (np.vstack([X, pca.inverse_transform(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_components': self.n_components,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class OUPS(OverSampling):
    """
    References:
        * BibTex::

            @article{oups,
                        title = "A priori synthetic over-sampling methods for
                                    increasing classification sensitivity in
                                    imbalanced data sets",
                        journal = "Expert Systems with Applications",
                        volume = "66",
                        pages = "124 - 135",
                        year = "2016",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2016.09.010",
                        author = "William A. Rivera and Petros Xanthopoulos",
                        keywords = "SMOTE, OUPS, Class imbalance,
                                    Classification"
                        }

    Notes:
        * In the description of the algorithm a fractional number p (j) is
            used to index a vector.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self, proportion=1.0, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if self.class_stats[self.min_label] < 2:
            message = ("The number of minority samples (%d) is not enough for"
                       " sampling")
            message = message % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + message)
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # extracting propensity scores
        lr = LogisticRegression(solver='lbfgs',
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)
        lr.fit(X, y)
        propensity = lr.predict_proba(X)
        propensity = propensity[:, np.where(
            lr.classes_ == self.min_label)[0][0]]

        # sorting indices according to propensity scores
        prop_sorted = sorted(zip(propensity, np.arange(
            len(propensity))), key=lambda x: -x[0])

        p = np.sum(y == self.maj_label)/np.sum(y == self.min_label)
        n = 0
        samples = []
        # implementing Algorithm 1 in the cited paper with some minor changes
        # to enable the proper sampling of p numbers
        while n < len(propensity) and len(samples) < n_to_sample:
            if (y[prop_sorted[n][1]] == self.min_label
                    and n < len(propensity) - 1):
                num = 1
                p_tmp = p
                while p_tmp > 0 and n + num < len(propensity):
                    if self.random_state.random_sample() < p_tmp:
                        samples.append(self.sample_between_points(
                            X[prop_sorted[n][1]], X[prop_sorted[n+num][1]]))
                    p_tmp = p_tmp - 1
                    num = num + 1
            n = n + 1

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_D(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{smote_d,
                            author="Torres, Fredy Rodr{\'i}guez
                            and Carrasco-Ochoa, Jes{\'u}s A.
                            and Mart{\'i}nez-Trinidad, Jos{\'e} Fco.",
                            editor="Mart{\'i}nez-Trinidad, Jos{\'e} Francisco
                            and Carrasco-Ochoa, Jes{\'u}s Ariel
                            and Ayala Ramirez, Victor
                            and Olvera-L{\'o}pez, Jos{\'e} Arturo
                            and Jiang, Xiaoyi",
                            title="SMOTE-D a Deterministic Version of SMOTE",
                            booktitle="Pattern Recognition",
                            year="2016",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="177--188",
                            isbn="978-3-319-39393-3"
                            }

    Notes:
        * Copying happens if two points are the neighbors of each other.
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self, proportion=1.0, k=3, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            k (int): number of neighbors in nearest neighbors component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, "k", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.k+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # extracting standard deviations of distances
        stds = np.std(dist[:, 1:], axis=1)

        # estimating sampling density
        if np.sum(stds) > 0:
            p_i = stds/np.sum(stds)
        else:
            _logger.warning(self.__class__.__name__ +
                            ": " + "zero distribution")
            return X.copy(), y.copy()

        # the other component of sampling density
        p_ij = dist[:, 1:]/np.sum(dist[:, 1:], axis=1)[:, None]

        # number of samples to generate between minority points
        counts_ij = n_to_sample*p_i[:, None]*p_ij

        # do the sampling
        samples = []
        for i in range(len(p_i)):
            for j in range(min([len(X_min)-1, self.k])):
                while counts_ij[i][j] > 0:
                    if self.random_state.random_sample() < counts_ij[i][j]:
                        translation = X_min[ind[i][j+1]] - X_min[i]
                        weight = counts_ij[i][j] + 1
                        samples.append(
                            X_min[i] + translation/counts_ij[i][j]+1)
                    counts_ij[i][j] = counts_ij[i][j] - 1

        if len(samples) > 0:
            return (np.vstack([X, np.vstack(samples)]),
                    np.hstack([y, np.repeat(self.min_label, len(samples))]))
        else:
            return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_PSO(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_pso,
                        title = "PSO-based method for SVM classification on
                                    skewed data sets",
                        journal = "Neurocomputing",
                        volume = "228",
                        pages = "187 - 197",
                        year = "2017",
                        note = "Advanced Intelligent Computing: Theory and
                                    Applications",
                        issn = "0925-2312",
                        doi = "https://doi.org/10.1016/j.neucom.2016.10.041",
                        author = "Jair Cervantes and Farid Garcia-Lamont and
                                    Lisbeth Rodriguez and Asdrúbal López and
                                    José Ruiz Castilla and Adrian Trueba",
                        keywords = "Skew data sets, SVM, Hybrid algorithms"
                        }

    Notes:
        * I find the description of the technique a bit confusing, especially
            on the bounds of the search space of velocities and positions.
            Equations 15 and 16 specify the lower and upper bounds, the lower
            bound is in fact a vector while the upper bound is a distance.
            I tried to implement something meaningful.
        * I also find the setting of accelerating constant 2.0 strange, most
            of the time the velocity will be bounded due to this choice.
        * Also, training and predicting probabilities with a non-linear
            SVM as the evaluation function becomes fairly expensive when the
            number of training vectors reaches a couple of thousands. To
            reduce computational burden, minority and majority vectors far
            from the other class are removed to reduce the size of both
            classes to a maximum of 500 samples. Generally, this shouldn't
            really affect the results as the technique focuses on the samples
            near the class boundaries.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_memetic,
                  OverSampling.cat_uses_classifier]

    def __init__(self,
                 k=3,
                 eps=0.05,
                 n_pop=10,
                 w=1.0,
                 c1=2.0,
                 c2=2.0,
                 num_it=10,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            k (int): number of neighbors in nearest neighbors component, this
                        is also the multiplication factor of minority support
                        vectors
            eps (float): use to specify the initially generated support
                            vectors along minority-majority lines
            n_pop (int): size of population
            w (float): intertia constant
            c1 (float): acceleration constant of local optimum
            c2 (float): acceleration constant of population optimum
            num_it (int): number of iterations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(num_it, "num_it", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.k = k
        self.eps = eps
        self.n_pop = n_pop
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.num_it = num_it
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'k': [3, 5, 7],
                                                    'eps': [0.05],
                                                    'n_pop': [5],
                                                    'w': [0.5, 1.0],
                                                    'c1': [1.0, 2.0],
                                                    'c2': [1.0, 2.0],
                                                    'num_it': [5]}, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # saving original dataset
        X_orig = X
        y_orig = y

        # scaling the records
        mms = MinMaxScaler()
        X_scaled = mms.fit_transform(X)

        # removing majority and minority samples far from the training data if
        # needed to increase performance
        performance_threshold = 500

        n_maj_to_remove = np.sum(
            y == self.maj_label) - performance_threshold
        if n_maj_to_remove > 0:
            # if majority samples are to be removed
            nn = NearestNeighbors(n_neighbors=1,
                                  n_jobs=self.n_jobs)
            nn.fit(X_scaled[y == self.min_label])
            dist, ind = nn.kneighbors(X_scaled)
            di = sorted([(dist[i][0], i)
                         for i in range(len(ind))], key=lambda x: x[0])
            to_remove = []
            # finding the proper number of samples farest from the minority
            # samples
            for i in reversed(range(len(di))):
                if y[di[i][1]] == self.maj_label:
                    to_remove.append(di[i][1])
                if len(to_remove) >= n_maj_to_remove:
                    break
            # removing the samples
            X_scaled = np.delete(X_scaled, to_remove, axis=0)
            y = np.delete(y, to_remove)

        n_min_to_remove = np.sum(
            y == self.min_label) - performance_threshold
        if n_min_to_remove > 0:
            # if majority samples are to be removed
            nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
            nn.fit(X_scaled[y == self.maj_label])
            dist, ind = nn.kneighbors(X_scaled)
            di = sorted([(dist[i][0], i)
                         for i in range(len(ind))], key=lambda x: x[0])
            to_remove = []
            # finding the proper number of samples farest from the minority
            # samples
            for i in reversed(range(len(di))):
                if y[di[i][1]] == self.min_label:
                    to_remove.append(di[i][1])
                if len(to_remove) >= n_min_to_remove:
                    break
            # removing the samples
            X_scaled = np.delete(X_scaled, to_remove, axis=0)
            y = np.delete(y, to_remove)

        # fitting SVM to extract initial support vectors
        svc = SVC(kernel='rbf', probability=True,
                  gamma='auto', random_state=self.random_state)
        svc.fit(X_scaled, y)

        # extracting the support vectors
        SV_min = np.array(
            [i for i in svc.support_ if y[i] == self.min_label])
        SV_maj = np.array(
            [i for i in svc.support_ if y[i] == self.maj_label])

        X_SV_min = X_scaled[SV_min]
        X_SV_maj = X_scaled[SV_maj]

        # finding nearest majority support vectors
        n_neighbors = min([len(X_SV_maj), self.k])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_SV_maj)
        dist, ind = nn.kneighbors(X_SV_min)

        # finding the initial particle and specifying the search space
        X_min_gen = []
        search_space = []
        init_velocity = []
        for i in range(len(SV_min)):
            for j in range(min([len(X_SV_maj), self.k])):
                min_vector = X_SV_min[i]
                maj_vector = X_SV_maj[ind[i][j]]
                # the upper bound of the search space if specified by the
                # closest majority support vector
                upper_bound = X_SV_maj[ind[i][0]]
                # the third element of the search space specification is
                # the distance of the vector and the closest
                # majority support vector, which specifies the radius of
                # the search
                norms = np.linalg.norm(min_vector - upper_bound)
                search_space.append([min_vector, maj_vector, norms])
                # initial particles
                X_min_gen.append(min_vector + self.eps *
                                 (maj_vector - min_vector))
                # initial velocities
                init_velocity.append(self.eps*(maj_vector - min_vector))

        X_min_gen = np.vstack(X_min_gen)
        init_velocity = np.vstack(init_velocity)

        # evaluates a specific particle
        def evaluate(X_train, y_train, X_test, y_test):
            """
            Trains support vector classifier and evaluates it

            Args:
                X_train (np.matrix): training vectors
                y_train (np.array): target labels
                X_test (np.matrix): test vectors
                y_test (np.array): test labels
            """
            svc.fit(X_train, y_train)
            y_pred = svc.predict_proba(X_test)[:, np.where(
                svc.classes_ == self.min_label)[0][0]]
            return roc_auc_score(y_test, y_pred)

        # initializing the particle swarm and the particle and population level
        # memory
        particle_swarm = [X_min_gen.copy() for _ in range(self.n_pop)]
        velocities = [init_velocity.copy() for _ in range(self.n_pop)]
        local_best = [X_min_gen.copy() for _ in range(self.n_pop)]
        local_best_scores = [0.0]*self.n_pop
        global_best = X_min_gen.copy()
        global_best_score = 0.0

        def evaluate_particle(X_scaled, p, y):
            X_extended = np.vstack([X_scaled, p])
            y_extended = np.hstack([y, np.repeat(self.min_label, len(p))])
            return evaluate(X_extended, y_extended, X_scaled, y)

        for i in range(self.num_it):
            _logger.info(self.__class__.__name__ + ": " + "Iteration %d" % i)
            # evaluate population
            scores = [evaluate_particle(X_scaled, p, y)
                      for p in particle_swarm]

            # update best scores
            for i, s in enumerate(scores):
                if s > local_best_scores[i]:
                    local_best_scores[i] = s
                    local_best[i] = particle_swarm[i]
                if s > global_best_score:
                    global_best_score = s
                    global_best = particle_swarm[i]

            # update velocities
            for i, p in enumerate(particle_swarm):
                term_0 = self.w*velocities[i]
                random_1 = self.random_state.random_sample()
                random_2 = self.random_state.random_sample()
                term_1 = self.c1*random_1*(local_best[i] - p)
                term_2 = self.c2*random_2*(global_best - p)

                velocities[i] = term_0 + term_1 + term_2

            # bound velocities according to search space constraints
            for v in velocities:
                for i in range(len(v)):
                    v_i_norm = np.linalg.norm(v[i])
                    if v_i_norm > search_space[i][2]/2.0:
                        v[i] = v[i]/v_i_norm*search_space[i][2]/2.0

            # update positions
            for i, p in enumerate(particle_swarm):
                particle_swarm[i] = particle_swarm[i] + velocities[i]

            # bound positions according to search space constraints
            for p in particle_swarm:
                for i in range(len(p)):
                    ss = search_space[i]

                    trans_vector = p[i] - ss[0]
                    trans_norm = np.linalg.norm(trans_vector)
                    normed_trans = trans_vector/trans_norm

                    if trans_norm > ss[2]:
                        p[i] = ss[0] + normed_trans*ss[2]

        X_ret = np.vstack([X_orig, mms.inverse_transform(global_best)])
        y_ret = np.hstack(
            [y_orig, np.repeat(self.min_label, len(global_best))])

        return (X_ret, y_ret)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'k': self.k,
                'eps': self.eps,
                'n_pop': self.n_pop,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
                'num_it': self.num_it,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class CURE_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{cure_smote,
                        author="Ma, Li
                        and Fan, Suohai",
                        title="CURE-SMOTE algorithm and hybrid algorithm for
                                feature selection and parameter optimization
                                based on random forests",
                        journal="BMC Bioinformatics",
                        year="2017",
                        month="Mar",
                        day="14",
                        volume="18",
                        number="1",
                        pages="169",
                        issn="1471-2105",
                        doi="10.1186/s12859-017-1578-z",
                        url="https://doi.org/10.1186/s12859-017-1578-z"
                        }

    Notes:
        * It is not specified how to determine the cluster with the
            "slowest growth rate"
        * All clusters can be removed as noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 n_clusters=5,
                 noise_th=2,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_clusters (int): number of clusters to generate
            noise_th (int): below this number of elements the cluster is
                                considered as noise
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_clusters, "n_clusters", 1)
        self.check_greater_or_equal(noise_th, "noise_th", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_clusters = n_clusters
        self.noise_th = noise_th
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_clusters': [5, 10, 15],
                                  'noise_th': [1, 3]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardizing the data
        mms = MinMaxScaler()
        X_scaled = mms.fit_transform(X)

        X_min = X_scaled[y == self.min_label]

        # initiating clustering
        clusters = [np.array([i]) for i in range(len(X_min))]
        dm = pairwise_distances(X_min)

        # setting the diagonal of the distance matrix to infinity
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # starting the clustering iteration
        iteration = 0
        while len(clusters) > self.n_clusters:
            iteration = iteration + 1

            # delete a cluster with slowest growth rate, determined by
            # the cluster size
            if iteration % self.n_clusters == 0:
                # extracting cluster sizes
                cluster_sizes = np.array([len(c) for c in clusters])
                # removing one of the clusters with the smallest size
                to_remove = np.where(cluster_sizes == np.min(cluster_sizes))[0]
                to_remove = self.random_state.choice(to_remove)
                del clusters[to_remove]
                # adjusting the distance matrix accordingly
                dm = np.delete(dm, to_remove, axis=0)
                dm = np.delete(dm, to_remove, axis=1)

            # finding the cluster pair with the smallest distance
            min_coord = np.where(dm == np.min(dm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # merging the clusters
            clusters[merge_a] = np.hstack(
                [clusters[merge_a], clusters[merge_b]])
            # removing one of them
            del clusters[merge_b]
            # adjusting the distances in the distance matrix
            dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]), axis=0)
            dm[:, merge_a] = dm[merge_a]
            # removing the row and column corresponding to one of
            # the merged clusters
            dm = np.delete(dm, merge_b, axis=0)
            dm = np.delete(dm, merge_b, axis=1)
            # updating the diagonal
            for i in range(len(dm)):
                dm[i, i] = np.inf

        # removing clusters declared as noise
        to_remove = []
        for i in range(len(clusters)):
            if len(clusters[i]) < self.noise_th:
                to_remove.append(i)
        clusters = [clusters[i]
                    for i in range(len(clusters)) if i not in to_remove]

        # all clusters can be noise
        if len(clusters) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "all clusters removed as noise")
            return X.copy(), y.copy()

        # generating samples
        samples = []
        for _ in range(n_to_sample):
            cluster_idx = self.random_state.randint(len(clusters))
            center = np.mean(X_min[clusters[cluster_idx]], axis=0)
            representative = X_min[self.random_state.choice(
                clusters[cluster_idx])]
            samples.append(self.sample_between_points(center, representative))

        return (np.vstack([X, mms.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_clusters': self.n_clusters,
                'noise_th': self.noise_th,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SOMO(OverSampling):
    """
    References:
        * BibTex::

            @article{somo,
                        title = "Self-Organizing Map Oversampling (SOMO) for
                                    imbalanced data set learning",
                        journal = "Expert Systems with Applications",
                        volume = "82",
                        pages = "40 - 52",
                        year = "2017",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2017.03.073",
                        author = "Georgios Douzas and Fernando Bacao"
                        }

    Notes:
        * It is not specified how to handle those cases when a cluster contains
            1 minority samples, the mean of within-cluster distances is set to
            100 in these cases.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 n_grid=10,
                 sigma=0.2,
                 learning_rate=0.5,
                 n_iter=100,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_grid (int): size of grid
            sigma (float): sigma of SOM
            learning_rate (float) learning rate of SOM
            n_iter (int): number of iterations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_grid, 'n_grid', 2)
        self.check_greater(sigma, 'sigma', 0)
        self.check_greater(learning_rate, 'learning_rate', 0)
        self.check_greater_or_equal(n_iter, 'n_iter', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_grid = n_grid
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_grid': [5, 9, 13],
                                  'sigma': [0.4],
                                  'learning_rate': [0.3, 0.5],
                                  'n_iter': [100]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        N_inter = n_to_sample/2
        N_intra = n_to_sample/2

        # training SOM
        som = minisom.MiniSom(self.n_grid,
                              self.n_grid,
                              len(X[0]),
                              sigma=self.sigma,
                              learning_rate=self.learning_rate,
                              random_seed=3)
        som.train_random(X, self.n_iter)

        # constructing the grid
        grid_min = {}
        grid_maj = {}
        for i in range(len(y)):
            tmp = som.winner(X[i])
            idx = (tmp[0], tmp[1])
            if idx not in grid_min:
                grid_min[idx] = []
            if idx not in grid_maj:
                grid_maj[idx] = []
            if y[i] == self.min_label:
                grid_min[idx].append(i)
            else:
                grid_maj[idx].append(i)

        # converting the grid to arrays
        for i in grid_min:
            grid_min[i] = np.array(grid_min[i])
        for i in grid_maj:
            grid_maj[i] = np.array(grid_maj[i])

        # filtering
        filtered = {}
        for i in grid_min:
            if i not in grid_maj:
                filtered[i] = True
            else:
                filtered[i] = (len(grid_maj[i]) + 1)/(len(grid_min[i])+1) < 1.0

        # computing densities
        densities = {}
        for i in filtered:
            if filtered[i]:
                if len(grid_min[i]) > 1:
                    paird = pairwise_distances(X[grid_min[i]])
                    densities[i] = len(grid_min[i])/np.mean(paird)**2
                else:
                    densities[i] = 10

        # all clusters can be filtered
        if len(densities) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "all clusters filtered")
            return X.copy(), y.copy()

        # computing neighbour densities, using 4 neighborhood
        neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        pair_densities = {}
        for i in densities:
            for n in neighbors:
                j = (i[0] + n[0], i[1] + n[1]),
                if j in densities:
                    pair_densities[(i, j)] = densities[i] + densities[j]

        # computing weights
        density_keys = list(densities.keys())
        density_vals = np.array(list(densities.values()))

        # determining pair keys and density values
        pair_keys = list(pair_densities.keys())
        pair_vals = np.array(list(pair_densities.values()))

        # determining densities
        density_vals = (1.0/density_vals)/np.sum(1.0/density_vals)
        pair_dens_vals = (1.0/pair_vals)/np.sum(1.0/pair_vals)

        # computing num of samples to generate
        if len(pair_vals) > 0:
            dens_num = N_intra
            pair_num = N_inter
        else:
            dens_num = N_inter + N_intra
            pair_num = 0

        # generating the samples according to the extracted distributions
        samples = []
        while len(samples) < dens_num:
            cluster_idx = density_keys[self.random_state.choice(
                np.arange(len(density_keys)), p=density_vals)]
            cluster = grid_min[cluster_idx]
            sample_a, sample_b = self.random_state.choice(cluster, 2)
            samples.append(self.sample_between_points(
                X[sample_a], X[sample_b]))

        while len(samples) < pair_num:
            idx = pair_keys[self.random_state.choice(
                np.arange(len(pair_keys)), p=pair_dens_vals)]
            cluster_a = grid_min[idx[0]]
            cluster_b = grid_min[idx[1]]
            X_a = X[self.random_state.choice(cluster_a)]
            X_b = X[self.random_state.choice(cluster_b)]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_grid': self.n_grid,
                'sigma': self.sigma,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ISOMAP_Hybrid(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{isomap_hybrid,
                             author = {Gu, Qiong and Cai, Zhihua and Zhu, Li},
                             title = {Classification of Imbalanced Data Sets by
                                        Using the Hybrid Re-sampling Algorithm
                                        Based on Isomap},
                             booktitle = {Proceedings of the 4th International
                                            Symposium on Advances in
                                            Computation and Intelligence},
                             series = {ISICA '09},
                             year = {2009},
                             isbn = {978-3-642-04842-5},
                             location = {Huangshi, China},
                             pages = {287--296},
                             numpages = {10},
                             doi = {10.1007/978-3-642-04843-2_31},
                             acmid = {1691478},
                             publisher = {Springer-Verlag},
                             address = {Berlin, Heidelberg},
                             keywords = {Imbalanced data set, Isomap, NCR,
                                            Smote, re-sampling},
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_components=3,
                 smote_n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            n_components (int): number of components
            smote_n_neighbors (int): number of neighbors in SMOTE sampling
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_components, "n_components", 1)
        self.check_greater_or_equal(smote_n_neighbors, "smote_n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.smote_n_neighbors = smote_n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_components': [2, 3, 4],
                                  'smote_n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        self.isomap = Isomap(n_neighbors=self.n_neighbors,
                             n_components=self.n_components,
                             n_jobs=self.n_jobs)

        X_trans = self.isomap.fit_transform(X, y)

        X_sm, y_sm = SMOTE(proportion=self.proportion,
                           n_neighbors=self.smote_n_neighbors,
                           n_jobs=self.n_jobs,
                           random_state=self.random_state).sample(X_trans, y)

        nc = NeighborhoodCleaningRule(n_jobs=self.n_jobs)
        return nc.remove_noise(X_sm, y_sm)

    def preprocessing_transform(self, X):
        """
        Transforms new data by the trained isomap

        Args:
            X (np.matrix): new data

        Returns:
            np.matrix: the transformed data
        """
        return self.isomap.transform(X)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_components': self.n_components,
                'smote_n_neighbors': self.smote_n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class CE_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ce_smote,
                                author={Chen, S. and Guo, G. and Chen, L.},
                                booktitle={2010 IEEE 24th International
                                            Conference on Advanced Information
                                            Networking and Applications
                                            Workshops},
                                title={A New Over-Sampling Method Based on
                                        Cluster Ensembles},
                                year={2010},
                                volume={},
                                number={},
                                pages={599-604},
                                keywords={data mining;Internet;pattern
                                            classification;pattern clustering;
                                            over sampling method;cluster
                                            ensembles;classification method;
                                            imbalanced data handling;CE-SMOTE;
                                            clustering consistency index;
                                            cluster boundary minority samples;
                                            imbalanced public data set;
                                            Mathematics;Computer science;
                                            Electronic mail;Accuracy;Nearest
                                            neighbor searches;Application
                                            software;Data mining;Conferences;
                                            Web sites;Information retrieval;
                                            classification;imbalanced data
                                            sets;cluster ensembles;
                                            over-sampling},
                                doi={10.1109/WAINA.2010.40},
                                ISSN={},
                                month={April}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 h=10,
                 k=5,
                 alpha=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            h (int): size of ensemble
            k (int): number of clusters/neighbors
            alpha (float): [0,1] threshold to select boundary samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(h, "h", 1)
        self.check_greater_or_equal(k, "k", 1)
        self.check_in_range(alpha, "alpha", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.h = h
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'h': [5, 10, 15],
                                  'k': [3, 5, 7],
                                  'alpha': [0.2, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # do the clustering and labelling
        d = len(X[0])
        labels = []
        for _ in range(self.h):
            f = self.random_state.randint(int(d/2), d)
            features = self.random_state.choice(np.arange(d), f)
            n_clusters = min([len(X), self.k])
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=self.random_state)
            kmeans.fit(X[:, features])
            labels.append(kmeans.labels_)

        # do the cluster matching, clustering 0 will be considered the one to
        # match the others to the problem of finding cluster matching is
        # basically the "assignment problem"
        base_label = 0
        for i in range(len(labels)):
            if not i == base_label:
                cost_matrix = np.zeros(shape=(self.k, self.k))
                for j in range(self.k):
                    mask_j = labels[base_label] == j
                    for k in range(self.k):
                        mask_k = labels[i] == k
                        mask_jk = np.logical_and(mask_j, mask_k)
                        cost_matrix[j, k] = np.sum(mask_jk)
                # solving the assignment problem
                row_ind, _ = soptimize.linear_sum_assignment(-cost_matrix)
                # doing the relabeling
                relabeling = labels[i].copy()
                for j in range(len(row_ind)):
                    relabeling[labels[i] == k] = j
                labels[i] = relabeling

        # compute clustering consistency index
        labels = np.vstack(labels)
        cci = np.apply_along_axis(lambda x: max(
            set(x.tolist()), key=x.tolist().count), 0, labels)
        cci = np.sum(labels == cci, axis=0)
        cci = cci/self.h

        # determining minority boundary samples
        P_boundary = X[np.logical_and(
            y == self.min_label, cci < self.alpha)]

        # there might be no boundary samples
        if len(P_boundary) <= 1:
            _logger.warning(self.__class__.__name__ + ": " + "empty boundary")
            return X.copy(), y.copy()

        # finding nearest neighbors of boundary samples
        n_neighbors = min([len(P_boundary), self.k])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs)
        nn.fit(P_boundary)
        dist, ind = nn.kneighbors(P_boundary)

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(ind))
            point_a = P_boundary[idx]
            point_b = P_boundary[self.random_state.choice(ind[idx][1:])]
            samples.append(self.sample_between_points(point_a, point_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'h': self.h,
                'k': self.k,
                'alpha': self.alpha,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Edge_Det_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{Edge_Det_SMOTE,
                            author={Kang, Y. and Won, S.},
                            booktitle={ICCAS 2010},
                            title={Weight decision algorithm for oversampling
                                    technique on class-imbalanced learning},
                            year={2010},
                            volume={},
                            number={},
                            pages={182-186},
                            keywords={edge detection;learning (artificial
                                        intelligence);weight decision
                                        algorithm;oversampling technique;
                                        class-imbalanced learning;class
                                        imbalanced data problem;edge
                                        detection algorithm;spatial space
                                        representation;Classification
                                        algorithms;Image edge detection;
                                        Training;Noise measurement;Glass;
                                        Training data;Machine learning;
                                        Imbalanced learning;Classification;
                                        Weight decision;Oversampling;
                                        Edge detection},
                            doi={10.1109/ICCAS.2010.5669889},
                            ISSN={},
                            month={Oct}}

    Notes:
        * This technique is very loosely specified.
    """

    categories = [OverSampling.cat_density_based,
                  OverSampling.cat_borderline,
                  OverSampling.cat_extensive]

    def __init__(self, proportion=1.0, k=5, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            k (int): number of neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, "k", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        d = len(X[0])
        X_min = X[y == self.min_label]

        # organizing class labels according to feature ranking
        magnitudes = np.zeros(len(X))
        for i in range(d):
            to_sort = zip(X[:, i], np.arange(len(X)), y)
            _, idx, label = zip(*sorted(to_sort, key=lambda x: x[0]))
            # extracting edge magnitudes in this dimension
            for j in range(1, len(idx)-1):
                magnitudes[idx[j]] = magnitudes[idx[j]] + \
                    (label[j-1] - label[j+1])**2

        # density estimation
        magnitudes = magnitudes[y == self.min_label]
        magnitudes = np.sqrt(magnitudes)
        magnitudes = magnitudes/np.sum(magnitudes)

        # fitting nearest neighbors models to minority samples
        n_neighbors = min([len(X_min), self.k+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.choice(np.arange(len(X_min)), p=magnitudes)
            X_a = X_min[idx]
            X_b = X_min[self.random_state.choice(ind[idx][1:])]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class CBSO(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{cbso,
                            author="Barua, Sukarna
                            and Islam, Md. Monirul
                            and Murase, Kazuyuki",
                            editor="Lu, Bao-Liang
                            and Zhang, Liqing
                            and Kwok, James",
                            title="A Novel Synthetic Minority Oversampling
                                    Technique for Imbalanced Data Set
                                    Learning",
                            booktitle="Neural Information Processing",
                            year="2011",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="735--744",
                            isbn="978-3-642-24958-7"
                            }

    Notes:
        * Clusters containing 1 element induce cloning of samples.
    """

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 C_p=1.3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            C_p (float): used to set the threshold of clustering
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater(C_p, "C_p", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.C_p = C_p
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'C_p': [0.8, 1.0, 1.3, 1.6]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model to find neighbors of minority points
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1,
                              n_jobs=self.n_jobs).fit(X)
        dist, ind = nn.kneighbors(X_min)

        # extracting the number of majority neighbors
        weights = [np.sum(y[ind[i][1:]] == self.maj_label)
                   for i in range(len(X_min))]
        # determine distribution of generating data
        weights = weights/np.sum(weights)

        # do the clustering
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs).fit(X_min)
        d_avg = np.mean(nn.kneighbors(X_min)[0][:, 1])
        T_h = d_avg*self.C_p

        # initiating clustering
        clusters = [np.array([i]) for i in range(len(X_min))]
        dm = pairwise_distances(X_min)

        # setting the diagonal of the distance matrix to infinity
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # starting the clustering iteration
        while True:
            # finding the cluster pair with the smallest distance
            min_coord = np.where(dm == np.min(dm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # check termination conditions
            if dm[merge_a, merge_b] > T_h or len(dm) == 1:
                break

            # merging the clusters
            clusters[merge_a] = np.hstack(
                [clusters[merge_a], clusters[merge_b]])
            # removing one of them
            del clusters[merge_b]
            # adjusting the distances in the distance matrix
            dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]), axis=0)
            dm[:, merge_a] = dm[merge_a]
            # removing the row and column corresponding to one of the
            # merged clusters
            dm = np.delete(dm, merge_b, axis=0)
            dm = np.delete(dm, merge_b, axis=1)
            # updating the diagonal
            for i in range(len(dm)):
                dm[i, i] = np.inf

        # extracting cluster labels
        labels = np.zeros(len(X_min)).astype(int)
        for i in range(len(clusters)):
            for j in clusters[i]:
                labels[j] = i

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)), p=weights)
            if len(clusters[labels[idx]]) <= 1:
                samples.append(X_min[idx])
                continue
            else:
                random_idx = self.random_state.choice(clusters[labels[idx]])
                while random_idx == idx:
                    random_idx = self.random_state.choice(
                        clusters[labels[idx]])
            samples.append(self.sample_between_points(
                X_min[idx], X_min[random_idx]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'C_p': self.C_p,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class E_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{e_smote,
                            author={Deepa, T. and Punithavalli, M.},
                            booktitle={2011 3rd International Conference on
                                        Electronics Computer Technology},
                            title={An E-SMOTE technique for feature selection
                                    in High-Dimensional Imbalanced Dataset},
                            year={2011},
                            volume={2},
                            number={},
                            pages={322-324},
                            keywords={bioinformatics;data mining;pattern
                                        classification;support vector machines;
                                        E-SMOTE technique;feature selection;
                                        high-dimensional imbalanced dataset;
                                        data mining;bio-informatics;dataset
                                        balancing;SVM classification;micro
                                        array dataset;Feature extraction;
                                        Genetic algorithms;Support vector
                                        machines;Data mining;Machine learning;
                                        Bioinformatics;Cancer;Imbalanced
                                        dataset;Featue Selection;E-SMOTE;
                                        Support Vector Machine[SVM]},
                            doi={10.1109/ICECTECH.2011.5941710},
                            ISSN={},
                            month={April}}

    Notes:
        * This technique is basically unreproducible. I try to implement
            something following the idea of applying some simple genetic
            algorithm for optimization.
        * In my best understanding, the technique uses evolutionary algorithms
            for feature selection and then applies vanilla SMOTE on the
            selected features only.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_memetic,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 min_features=2,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in the nearest neighbors
                                component
            min_features (int): minimum number of features
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(min_features, "min_features", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.min_features = min_features
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'min_features': [1, 2, 3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        min_features = min(self.min_features, len(X[0]))

        if len(X) < 800:
            classifier = SVC(gamma='auto', random_state=self.random_state)
        else:
            classifier = DecisionTreeClassifier(
                max_depth=4, random_state=self.random_state)

        # parameters of the evolutionary algorithm
        n_generations = 50
        n_population = 5

        # creating initial mask
        mask = self.random_state.choice([True, False], len(X[0]), replace=True)
        # fixing if the mask doesn't contain any features
        if np.sum(mask) == 0:
            mask[self.random_state.randint(len(mask))] = True

        def crossover(mask_a, mask_b):
            """
            Crossover operation for two masks

            Args:
                mask_a (np.array): binary mask 1
                mask_b (np.array): binary mask 2

            Returns:
                np.array: the result of crossover
            """
            mask = mask_a.copy()
            for i in range(len(mask_b)):
                if self.random_state.randint(0, 2) == 0:
                    mask[i] = mask_b[i]

            while np.sum(mask) < min_features:
                mask[self.random_state.randint(len(mask))] = True

            return mask

        def mutate(mask_old):
            """
            Mutation operation for a mask

            Args:
                mask_old (np.array): binary mask

            Returns:
                np.array: the result of mutation
            """
            mask = mask_old.copy()
            for i in range(len(mask)):
                if self.random_state.randint(0, 2) == 0:
                    mask[i] = not mask[i]

            while np.sum(mask) < min_features:
                mask[self.random_state.randint(len(mask))] = True

            return mask

        # generating initial population
        population = [[0, mask.copy()] for _ in range(n_population)]
        for _ in range(n_generations):
            # in each generation
            for _ in range(n_population):
                # for each element of a population
                if self.random_state.randint(0, 2) == 0:
                    # crossover
                    i_0 = self.random_state.randint(n_population)
                    i_1 = self.random_state.randint(n_population)
                    mask = crossover(population[i_0][1], population[i_1][1])
                else:
                    # mutation
                    idx = self.random_state.randint(n_population)
                    mask = mutate(population[idx][1])
                # evaluation
                message = "evaluating mask selection with features %d/%d"
                message = message % (np.sum(mask), len(mask))
                _logger.info(self.__class__.__name__ + ": " + message)
                classifier.fit(X[:, mask], y)
                score = np.sum(y == classifier.predict(X[:, mask]))/len(y)
                # appending the result to the population
                population.append([score, mask])
            # sorting the population in a reversed order and keeping the
            # elements with the highest scores
            population = sorted(population, key=lambda x: -x[0])[:n_population]

        self.mask = population[0][1]
        # resampling the population in the given dimensions

        smote = SMOTE(self.proportion,
                      self.n_neighbors,
                      n_jobs=self.n_jobs,
                      random_state=self.random_state)

        return smote.sample(X[:, self.mask], y)

    def preprocessing_transform(self, X):
        """
        Transform new data by the learnt transformation

        Args:
            X (np.matrix): new data

        Returns:
            np.matrix: transformed data
        """
        return X[:, self.mask]

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'min_features': self.min_features,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class DBSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{dbsmote,
                        author="Bunkhumpornpat, Chumphol
                        and Sinapiromsaran, Krung
                        and Lursinsap, Chidchanok",
                        title="DBSMOTE: Density-Based Synthetic Minority
                                Over-sampling TEchnique",
                        journal="Applied Intelligence",
                        year="2012",
                        month="Apr",
                        day="01",
                        volume="36",
                        number="3",
                        pages="664--684",
                        issn="1573-7497",
                        doi="10.1007/s10489-011-0287-y",
                        url="https://doi.org/10.1007/s10489-011-0287-y"
                        }

    Notes:
        * Standardization is needed to use absolute eps values.
        * The clustering is likely to identify all instances as noise, fixed
            by recursive call with increaseing eps.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based]

    def __init__(self,
                 proportion=1.0,
                 eps=0.8,
                 min_samples=3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            eps (float): eps paramter of DBSCAN
            min_samples (int): min_samples paramter of DBSCAN
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(min_samples, "min_samples", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'eps': [0.5, 0.8, 1.2],
                                  'min_samples': [1, 3, 5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        ss = StandardScaler().fit(X)
        X_ss = ss.transform(X)

        # doing the clustering using DBSCAN
        X_min = X_ss[y == self.min_label]
        db = DBSCAN(self.eps, self.min_samples, n_jobs=self.n_jobs).fit(X_min)
        labels = db.labels_
        num_labels = np.max(labels)+1

        if num_labels == 0:
            # adjusting the parameters if no clusters were identified
            message = ("Number of clusters is 0, trying to increase eps and "
                       "decrease min_samples")
            _logger.info(self.__class__.__name__ + ": " + message)
            if self.eps >= 2 or self.min_samples <= 2:
                message = ("Number of clusters is 0, can't adjust parameters "
                           "further")
                _logger.info(self.__class__.__name__ + ": " + message)
                return X.copy(), y.copy()
            else:
                return DBSMOTE(proportion=self.proportion,
                               eps=self.eps*1.5,
                               min_samples=self.min_samples-1,
                               n_jobs=self.n_jobs,
                               random_state=self.random_state).sample(X, y)

        # determining cluster size distribution
        clusters = [np.where(labels == i)[0] for i in range(num_labels)]
        cluster_sizes = np.array([np.sum(labels == i)
                                  for i in range(num_labels)])
        cluster_dist = cluster_sizes/np.sum(cluster_sizes)

        # Bellman-Ford algorithm, inspired by
        # https://gist.github.com/joninvski/701720
        def initialize(graph, source):
            """
            Initializes shortest path algorithm.

            Args:
                graph (dict): graph in dictionary representation
                source (key): source node

            Returns:
                dict, dict: initialized distance and path dictionaries
            """
            d = {}
            p = {}
            for node in graph:
                d[node] = float('Inf')
                p[node] = None
            d[source] = 0
            return d, p

        def relax(u, v, graph, d, p):
            """
            Checks if shorter path exists.

            Args:
                u (key): key of a node
                v (key): key of another node
                graph (dict): the graph object
                d (dict): the distances dictionary
                p (dict): the paths dictionary
            """
            if d[v] > d[u] + graph[u][v]:
                d[v] = d[u] + graph[u][v]
                p[v] = u

        def bellman_ford(graph, source):
            """
            Main entry point of the Bellman-Ford algorithm

            Args:
                graph (dict): a graph in dictionary representation
                source (key): the key of the source node
            """
            d, p = initialize(graph, source)
            for i in range(len(graph)-1):
                for u in graph:
                    for v in graph[u]:
                        relax(u, v, graph, d, p)
            for u in graph:
                for v in graph[u]:
                    assert d[v] <= d[u] + graph[u][v]
            return d, p

        # extract graphs and center-like objects
        graphs = []
        centroid_indices = []
        shortest_paths = []
        for c in range(num_labels):
            # extracting the cluster elements
            cluster = X_min[clusters[c]]
            # initializing the graph object
            graph = {}
            for i in range(len(cluster)):
                graph[i] = {}

            # fitting nearest neighbors model to the cluster elements
            nn = NearestNeighbors(n_neighbors=len(cluster), n_jobs=self.n_jobs)
            nn.fit(cluster)
            dist, ind = nn.kneighbors(cluster)

            # extracting graph edges according to directly density reachabality
            # definition
            for i in range(len(cluster)):
                n = min([len(cluster), (self.min_samples + 1)])
                index_set = ind[i][1:n]
                for j in range(len(cluster)):
                    if j in index_set and dist[i][ind[i] == j][0] < self.eps:
                        graph[i][j] = dist[i][ind[i] == j][0]
            graphs.append(graph)
            # finding the index of the center like object
            centroid_ind = nn.kneighbors(
                np.mean(cluster, axis=0).reshape(1, -1))[1][0][0]
            centroid_indices.append(centroid_ind)
            # extracting shortest paths from centroid object
            shortest_paths.append(bellman_ford(graph, centroid_ind))

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(
                np.arange(len(clusters)), p=cluster_dist)
            cluster = X_min[clusters[cluster_idx]]
            idx = self.random_state.choice(range(len(clusters[cluster_idx])))

            # executing shortest path algorithm
            distances, parents = shortest_paths[cluster_idx]

            # extracting path
            path = [idx]
            while not parents[path[-1]] is None:
                path.append(parents[path[-1]])

            if len(path) == 1:
                # if the center like object is selected
                samples.append(cluster[path[0]])
            elif len(path) == 2:
                # if the path consists of 1 edge
                X_a = cluster[path[0]]
                X_b = cluster[path[1]]
                sample = self.sample_between_points_componentwise(X_a, X_b)
                samples.append(sample)
            else:
                # if the path consists of at least two edges
                random_vertex = self.random_state.randint(len(path)-1)
                X_a = cluster[path[random_vertex]]
                X_b = cluster[path[random_vertex + 1]]
                sample = self.sample_between_points_componentwise(X_a, X_b)
                samples.append(sample)

        return (np.vstack([X, ss.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'eps': self.eps,
                'min_samples': self.min_samples,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ASMOBD(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{asmobd,
                            author={Senzhang Wang and Zhoujun Li and Wenhan
                                    Chao and Qinghua Cao},
                            booktitle={The 2012 International Joint Conference
                                        on Neural Networks (IJCNN)},
                            title={Applying adaptive over-sampling technique
                                    based on data density and cost-sensitive
                                    SVM to imbalanced learning},
                            year={2012},
                            volume={},
                            number={},
                            pages={1-8},
                            doi={10.1109/IJCNN.2012.6252696},
                            ISSN={2161-4407},
                            month={June}}

    Notes:
        * In order to use absolute thresholds, the data is standardized.
        * The technique has many parameters, not easy to find the right
            combination.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 min_samples=3,
                 eps=0.8,
                 eta=0.5,
                 T_1=1.0,
                 T_2=1.0,
                 t_1=4.0,
                 t_2=4.0,
                 a=0.05,
                 smoothing='linear',
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            min_samples (int): parameter of OPTICS
            eps (float): parameter of OPTICS
            eta (float): tradeoff paramter
            T_1 (float): noise threshold (see paper)
            T_2 (float): noise threshold (see paper)
            t_1 (float): noise threshold (see paper)
            t_2 (float): noise threshold (see paper)
            a (float): smoothing factor (see paper)
            smoothing (str): 'sigmoid'/'linear'
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(min_samples, "min_samples", 1)
        self.check_greater(eps, "eps", 0)
        self.check_in_range(eta, "eta", [0, 1])
        self.check_greater(T_1, "T_1", 0)
        self.check_greater(T_2, "T_2", 0)
        self.check_greater(t_1, "t_1", 0)
        self.check_greater(t_2, "t_2", 0)
        self.check_greater(a, "a", 0)
        self.check_isin(smoothing, "smoothing", ['sigmoid', 'linear'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.min_samples = min_samples
        self.eps = eps
        self.eta = eta
        self.T_1 = T_1
        self.T_2 = T_2
        self.t_1 = t_1
        self.t_2 = t_2
        self.a = a
        self.smoothing = smoothing
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'min_samples': [3],
                                  'eps': [0.3],
                                  'eta': [0.5],
                                  'T_1': [0.7, 1.0, 1.4],
                                  'T_2': [0.7, 1.0, 1.4],
                                  't_1': [4.0],
                                  't_2': [4.0],
                                  'a': [0.05, 0.1],
                                  'smoothing': ['sigmoid', 'linear']}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardizing the data to enable using absolute thresholds
        ss = StandardScaler().fit(X)
        X_ss = ss.transform(X)

        X_min = X_ss[y == self.min_label]

        # executing the optics algorithm
        min_samples = min([len(X_min)-1, self.min_samples])
        o = OPTICS(min_samples=min_samples,
                   max_eps=self.eps,
                   n_jobs=self.n_jobs)
        o.fit(X_min)
        cd = o.core_distances_
        r = o.reachability_

        # identifying noise
        noise = np.logical_and(cd > self.T_1, r > self.T_2)

        # fitting nearest neighbors models to identify the number of majority
        # samples in local environments
        nn = NearestNeighbors(n_neighbors=self.min_samples, n_jobs=self.n_jobs)
        nn.fit(X_ss)
        n_majs = []
        ratio = []
        for i in range(len(X_min)):
            ind = nn.radius_neighbors(X_min[i].reshape(
                1, -1), radius=cd[i], return_distance=False)[0]
            n_maj = np.sum(y[ind] == self.maj_label)/len(ind)
            n_majs.append(n_maj)
            n_min = len(ind) - n_maj - 1
            if n_min == 0:
                ratio.append(np.inf)
            else:
                ratio.append(n_maj/n_min)

        n_maj = np.array(n_maj)
        ratio = np.array(ratio)

        # second constraint on noise
        noise_2 = np.logical_and(cd > np.mean(
            cd)*self.t_1, r > np.mean(r)*self.t_2)

        # calculating density according to the smoothing function specified
        if self.smoothing == 'sigmoid':
            balance_ratio = np.abs(2.0/(1.0 + np.exp(-self.a*ratio[i])) - 1.0)
            df = self.eta*cd + (1.0 - self.eta)*n_maj - balance_ratio
        else:
            df = self.eta*(self.eta*cd + (1.0 - self.eta)*n_maj) + \
                (1 - self.eta)*len(X_min)/n_to_sample

        # unifying the conditions on noise
        not_noise = np.logical_not(np.logical_or(noise, noise_2))

        # checking if there are not noise samples remaining
        if np.sum(not_noise) == 0:
            message = ("All minority samples found to be noise, increasing"
                       "noise thresholds")
            _logger.info(self.__class__.__name__ + ": " + message)

            return ASMOBD(proportion=self.proportion,
                          min_samples=self.min_samples,
                          eps=self.eps,
                          eta=self.eta,
                          T_1=self.T_1*1.5,
                          T_2=self.T_2*1.5,
                          t_1=self.t_1*1.5,
                          t_2=self.t_2*1.5,
                          a=self.a,
                          smoothing=self.smoothing,
                          n_jobs=self.n_jobs,
                          random_state=self.random_state).sample(X, y)

        # removing noise and adjusting the density factors accordingly
        X_min_not_noise = X_min[not_noise]

        # checking if there are not-noisy samples
        if len(X_min_not_noise) <= 2:
            _logger.warning(self.__class__.__name__ + ": " +
                            "no not-noise minority sample remained")
            return X.copy(), y.copy()

        df = np.delete(df, np.where(np.logical_not(not_noise))[0])
        density = df/np.sum(df)

        # fitting nearest neighbors model to non-noise minority samples
        n_neighbors = min([len(X_min_not_noise), self.min_samples + 1])
        nn_not_noise = NearestNeighbors(n_neighbors=n_neighbors,
                                        n_jobs=self.n_jobs)
        nn_not_noise.fit(X_min_not_noise)
        dist, ind = nn_not_noise.kneighbors(X_min_not_noise)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min_not_noise)),
                                           p=density)
            random_neighbor_idx = self.random_state.choice(ind[idx][1:])
            X_a = X_min_not_noise[idx]
            X_b = X_min_not_noise[random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, ss.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'min_samples': self.min_samples,
                'eps': self.eps,
                'eta': self.eta,
                'T_1': self.T_1,
                'T_2': self.T_2,
                't_1': self.t_1,
                't_2': self.t_2,
                'a': self.a,
                'smoothing': self.smoothing,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Assembled_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{assembled_smote,
                            author={Zhou, B. and Yang, C. and Guo, H. and
                                        Hu, J.},
                            booktitle={The 2013 International Joint Conference
                                        on Neural Networks (IJCNN)},
                            title={A quasi-linear SVM combined with assembled
                                    SMOTE for imbalanced data classification},
                            year={2013},
                            volume={},
                            number={},
                            pages={1-7},
                            keywords={approximation theory;interpolation;
                                        pattern classification;sampling
                                        methods;support vector machines;trees
                                        (mathematics);quasilinear SVM;
                                        assembled SMOTE;imbalanced dataset
                                        classification problem;oversampling
                                        method;quasilinear kernel function;
                                        approximate nonlinear separation
                                        boundary;mulitlocal linear boundaries;
                                        interpolation;data distribution
                                        information;minimal spanning tree;
                                        local linear partitioning method;
                                        linear separation boundary;synthetic
                                        minority class samples;oversampled
                                        dataset classification;standard SVM;
                                        composite quasilinear kernel function;
                                        artificial data datasets;benchmark
                                        datasets;classification performance
                                        improvement;synthetic minority
                                        over-sampling technique;Support vector
                                        machines;Kernel;Merging;Standards;
                                        Sociology;Statistics;Interpolation},
                            doi={10.1109/IJCNN.2013.6707035},
                            ISSN={2161-4407},
                            month={Aug}}

    Notes:
        * Absolute value of the angles extracted should be taken.
            (implemented this way)
        * It is not specified how many samples are generated in the various
            clusters.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_borderline,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 pop=2,
                 thres=0.3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            pop (int): lower threshold on cluster sizes
            thres (float): threshold on angles
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(pop, "pop", 1)
        self.check_in_range(thres, "thres", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.pop = pop
        self.thres = thres
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'pop': [2, 4, 5],
                                  'thres': [0.1, 0.3, 0.5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        # finding the set of border and non-border minority elements
        n_min_neighbors = [np.sum(y[ind[i]] == self.min_label)
                           for i in range(len(ind))]
        border_mask = np.logical_not(np.array(n_min_neighbors) == n_neighbors)
        X_border = X_min[border_mask]
        X_non_border = X_min[np.logical_not(border_mask)]

        if len(X_border) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "X_border is empty")
            return X.copy(), y.copy()

        # initializing clustering
        clusters = [np.array([i]) for i in range(len(X_border))]
        dm = pairwise_distances(X_border)
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # do the clustering
        while len(dm) > 1 and np.min(dm) < np.inf:
            # extracting coordinates of clusters with the minimum distance
            min_coord = np.where(dm == np.min(dm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # checking the size of clusters to see if they should be merged
            if (len(clusters[merge_a]) < self.pop
                    or len(clusters[merge_b]) < self.pop):
                # if both clusters are small, do the merge
                clusters[merge_a] = np.hstack([clusters[merge_a],
                                               clusters[merge_b]])
                del clusters[merge_b]
                # update the distance matrix accordingly
                dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]),
                                     axis=0)
                dm[:, merge_a] = dm[merge_a]
                # remove columns
                dm = np.delete(dm, merge_b, axis=0)
                dm = np.delete(dm, merge_b, axis=1)
                # fix the diagonal entries
                for i in range(len(dm)):
                    dm[i, i] = np.inf
            else:
                # otherwise find principal directions
                pca_a = PCA(n_components=1).fit(X_border[clusters[merge_a]])
                pca_b = PCA(n_components=1).fit(X_border[clusters[merge_b]])
                # extract the angle of principal directions
                numerator = np.dot(pca_a.components_[0], pca_b.components_[0])
                denominator = np.linalg.norm(pca_a.components_[0])
                denominator *= np.linalg.norm(pca_b.components_[0])
                angle = abs(numerator/denominator)
                # check if angle if angle is above a specific threshold
                if angle > self.thres:
                    # do the merge
                    clusters[merge_a] = np.hstack([clusters[merge_a],
                                                   clusters[merge_b]])
                    del clusters[merge_b]
                    # update the distance matrix acoordingly
                    dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]),
                                         axis=0)
                    dm[:, merge_a] = dm[merge_a]
                    # remove columns
                    dm = np.delete(dm, merge_b, axis=0)
                    dm = np.delete(dm, merge_b, axis=1)
                    # fixing the digaonal entries
                    for i in range(len(dm)):
                        dm[i, i] = np.inf
                else:
                    # changing the distance of clusters to fininte
                    dm[merge_a, merge_b] = np.inf
                    dm[merge_b, merge_a] = np.inf

        # extract vectors belonging to the various clusters
        vectors = [X_border[c] for c in clusters if len(c) > 0]
        # adding non-border samples
        if len(X_non_border) > 0:
            vectors.append(X_non_border)

        # extract cluster sizes and calculating point distribution in clusters
        # the last element of the clusters is the set of non-border xamples
        cluster_sizes = np.array([len(v) for v in vectors])
        densities = cluster_sizes/np.sum(cluster_sizes)

        # extracting nearest neighbors in clusters
        def fit_knn(vectors):
            n_neighbors = min([self.n_neighbors + 1, len(vectors)])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            return nn.fit(vectors).kneighbors(vectors)

        nns = [fit_knn(v) for v in vectors]

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(len(vectors), p=densities)
            len_cluster = len(vectors[cluster_idx])
            sample_idx = self.random_state.choice(np.arange(len_cluster))

            if len_cluster > 1:
                choose_from = nns[cluster_idx][1][sample_idx][1:]
                random_neighbor_idx = self.random_state.choice(choose_from)
            else:
                random_neighbor_idx = sample_idx

            X_a = vectors[cluster_idx][sample_idx]
            X_b = vectors[cluster_idx][random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'pop': self.pop,
                'thres': self.thres,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SDSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{sdsmote,
                            author={Li, K. and Zhang, W. and Lu, Q. and
                                        Fang, X.},
                            booktitle={2014 International Conference on
                                        Identification, Information and
                                        Knowledge in the Internet of
                                        Things},
                            title={An Improved SMOTE Imbalanced Data
                                    Classification Method Based on Support
                                    Degree},
                            year={2014},
                            volume={},
                            number={},
                            pages={34-38},
                            keywords={data mining;pattern classification;
                                        sampling methods;improved SMOTE
                                        imbalanced data classification
                                        method;support degree;data mining;
                                        class distribution;imbalanced
                                        data-set classification;over sampling
                                        method;minority class sample
                                        generation;minority class sample
                                        selection;minority class boundary
                                        sample identification;Classification
                                        algorithms;Training;Bagging;Computers;
                                        Testing;Algorithm design and analysis;
                                        Data mining;Imbalanced data-sets;
                                        Classification;Boundary sample;Support
                                        degree;SMOTE},
                            doi={10.1109/IIKI.2014.14},
                            ISSN={},
                            month={Oct}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_borderline]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model to find closest majority points to
        # minority samples
        nn = NearestNeighbors(n_neighbors=len(X_maj), n_jobs=self.n_jobs)
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)

        # calculating the sum according to S3 in the paper
        S_i = np.sum(dist, axis=1)
        # calculating average distance according to S5
        S = np.sum(S_i)
        S_ave = S/(len(X_min)*len(X_maj))

        # calculate support degree
        def support_degree(x):
            return len(nn.radius_neighbors(x.reshape(1, -1),
                                           S_ave,
                                           return_distance=False))

        k = np.array([support_degree(X_min[i]) for i in range(len(X_min))])
        density = k/np.sum(k)

        # fitting nearest neighbors model to minority samples to run
        # SMOTE-like sampling
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(density)), p=density)
            random_neighbor_idx = self.random_state.choice(ind[idx][1:])
            X_a = X_min[idx]
            X_b = X_min[random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class DSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{dsmote,
                            author={Mahmoudi, S. and Moradi, P. and Akhlaghian,
                                    F. and Moradi, R.},
                            booktitle={2014 4th International Conference on
                                        Computer and Knowledge Engineering
                                        (ICCKE)},
                            title={Diversity and separable metrics in
                                    over-sampling technique for imbalanced
                                    data classification},
                            year={2014},
                            volume={},
                            number={},
                            pages={152-158},
                            keywords={learning (artificial intelligence);
                                        pattern classification;sampling
                                        methods;diversity metric;separable
                                        metric;over-sampling technique;
                                        imbalanced data classification;
                                        class distribution techniques;
                                        under-sampling technique;DSMOTE method;
                                        imbalanced learning problem;diversity
                                        measure;separable measure;Iran
                                        University of Medical Science;UCI
                                        dataset;Accuracy;Classification
                                        algorithms;Vectors;Educational
                                        institutions;Euclidean distance;
                                        Data mining;Diversity measure;
                                        Separable Measure;Over-Sampling;
                                        Imbalanced Data;Classification
                                        problems},
                            doi={10.1109/ICCKE.2014.6993409},
                            ISSN={},
                            month={Oct}}

    Notes:
        * The method is highly inefficient when the number of minority samples
            is high, time complexity is O(n^3), with 1000 minority samples it
            takes about 1e9 objective function evaluations to find 1 new sample
            points. Adding 1000 samples would take about 1e12 evaluations of
            the objective function, which is unfeasible. We introduce a new
            parameter, n_step, and during the search for the new sample at
            most n_step combinations of minority samples are tried.
        * Abnormality of minority points is defined in the paper as
            D_maj/D_min, high abnormality  means that the minority point is
            close to other minority points and very far from majority points.
            This is definitely not abnormality,
            I have implemented the opposite.
        * Nothing ensures that the fisher statistics and the variance from
            the geometric mean remain comparable, which might skew the
            optimization towards one of the sub-objectives.
        * MinMax normalization doesn't work, each attribute will have a 0
            value, which will make the geometric mean of all attribute 0.
    """

    categories = [OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 rate=0.1,
                 n_step=50,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            rate (float): [0,1] rate of minority samples to turn into majority
            n_step (int): number of random configurations to check for new
                                samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(rate, "rate", [0, 1])
        self.check_greater_or_equal(n_step, "n_step", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.rate = rate
        self.n_step = n_step
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'rate': [0.1, 0.2],
                                  'n_step': [50]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        mms = MinMaxScaler(feature_range=(1e-6, 1.0 - 1e-6))
        X = mms.fit_transform(X)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model
        nn = NearestNeighbors(n_neighbors=len(X_maj))
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)

        # compute mean distances, the D_min is compenstaed for taking into
        # consideration self-distances in the mean
        D_maj = np.mean(dist, axis=1)
        D_min = np.mean(pairwise_distances(X_min), axis=1) * \
            len(X_min)/(len(X_min)-1)

        # computing degree of abnormality
        abnormality = D_min/D_maj

        # sorting minority indices in decreasing order by abnormality
        to_sort = zip(abnormality, np.arange(len(abnormality)))
        abnormality, indices = zip(*sorted(to_sort, key=lambda x: -x[0]))
        rate = int(self.rate*len(abnormality))

        if rate > 0:
            # moving the most abnormal points to the majority class
            X_maj = np.vstack([X_maj, X_min[np.array(indices[:rate])]])
            # removing the most abnormal points form the minority class
            X_min = np.delete(X_min, indices[:rate], axis=0)

        # computing the mean and variance of points in the majority class
        var_maj = np.mean(np.var(X_maj, axis=0))
        mean_maj = np.mean(X_maj)

        # this is the original objective function, however, using this
        # is very inefficient if the number of records increases above
        # approximately 1000
        # def objective(X):
        #    """
        #    The objective function to be maximized
        #
        #    Args:
        #        X (np.matrix): dataset
        #
        #    Returns:
        #        float: the value of the objective function
        #    """
        #    gm= gmean(X, axis= 0)
        #    gdiv= np.mean(np.linalg.norm(X - gm, axis= 1))
        #    fisher= (np.mean(X) - mean_maj)**2/(np.mean(np.var(X, axis= 0)) \
        #                + var_maj)
        #    return gdiv + fisher

        # in order to make the code more efficient, we do maintain some
        # variables containing the main componentes of the objective function
        # and apply only small corrections based on the new values being added
        # the effect should be identical

        # records the sum of logarithms in X_min, used to compute the geometric
        # mean
        min_log_sum = np.sum(np.log(X_min), axis=0)
        # contains the sum of values in X_min, coordinatewise
        min_sum = np.sum(X_min, axis=0)
        # contains the squares of sums of values in X_min, coordinatewise
        min_sum2 = np.sum(X_min**2, axis=0)
        # contains the sum of all numbers in X_min
        min_all_sum = np.sum(X_min)

        min_norm = np.linalg.norm(X_min)**2

        # do the sampling
        n_added = 0
        while n_added < n_to_sample:
            best_candidate = None
            highest_score = 0.0
            # we try n_step combinations of minority samples
            len_X = len(X_min)
            n_steps = min([len_X*(len_X-1)*(len_X-2), self.n_step])
            for _ in range(n_steps):
                i, j, k = self.random_state.choice(np.arange(len_X),
                                                   3,
                                                   replace=False)
                gm = gmean(X_min[np.array([i, j, k])], axis=0)

                # computing the new objective function for the new point (gm)
                #  added
                new_X_min = np.vstack([X_min, gm])

                # updating the components of the objective function
                new_min_log_sum = min_log_sum + np.log(gm)
                new_min_sum = min_sum + gm
                new_min_sum2 = min_sum2 + gm**2
                new_min_all_sum = min_all_sum + np.sum(gm)

                # computing mean, var, gmean and mean of all elements with
                # the new sample (gm)
                new_min_mean = new_min_sum/(len(new_X_min))
                new_min_var = new_min_sum2/(len(new_X_min)) - new_min_mean**2
                new_min_gmean = np.exp(new_min_log_sum/(len(new_X_min)))
                new_min_all_n = (len(new_X_min))*len(X_min[0])
                new_min_all_mean = new_min_all_sum / new_min_all_n

                new_min_norm = min_norm + np.linalg.norm(gm)

                # computing the new objective function value
                inner_prod = np.dot(new_X_min, new_min_gmean)
                gmean_norm = np.linalg.norm(new_min_gmean)**2
                term_sum = new_min_norm - 2*inner_prod + gmean_norm
                new_gdiv = np.mean(np.sqrt(term_sum))

                fisher_numerator = (new_min_all_mean - mean_maj)**2
                fisher_denominator = np.mean(new_min_var) + var_maj
                new_fisher = fisher_numerator / fisher_denominator

                score = new_gdiv + new_fisher

                # evaluate the objective function
                # score= objective(np.vstack([X_min, gm]))
                # check if the score is better than the best so far
                if score > highest_score:
                    highest_score = score
                    best_candidate = gm
                    cand_min_log_sum = new_min_log_sum
                    cand_min_sum = new_min_sum
                    cand_min_sum2 = new_min_sum2
                    cand_min_all_sum = new_min_all_sum
                    cand_min_norm = new_min_norm

            # add the best candidate to the minority samples
            X_min = np.vstack([X_min, best_candidate])
            n_added = n_added + 1

            min_log_sum = cand_min_log_sum
            min_sum = cand_min_sum
            min_sum2 = cand_min_sum2
            min_all_sum = cand_min_all_sum
            min_norm = cand_min_norm

        return (mms.inverse_transform(np.vstack([X_maj, X_min])),
                np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'rate': self.rate,
                'n_step': self.n_step,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class G_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{g_smote,
                            author={Sandhan, T. and Choi, J. Y.},
                            booktitle={2014 22nd International Conference on
                                        Pattern Recognition},
                            title={Handling Imbalanced Datasets by Partially
                                    Guided Hybrid Sampling for Pattern
                                    Recognition},
                            year={2014},
                            volume={},
                            number={},
                            pages={1449-1453},
                            keywords={Gaussian processes;learning (artificial
                                        intelligence);pattern classification;
                                        regression analysis;sampling methods;
                                        support vector machines;imbalanced
                                        datasets;partially guided hybrid
                                        sampling;pattern recognition;real-world
                                        domains;skewed datasets;dataset
                                        rebalancing;learning algorithm;
                                        extremely low minority class samples;
                                        classification tasks;extracted hidden
                                        patterns;support vector machine;
                                        logistic regression;nearest neighbor;
                                        Gaussian process classifier;Support
                                        vector machines;Proteins;Pattern
                                        recognition;Kernel;Databases;Gaussian
                                        processes;Vectors;Imbalanced dataset;
                                        protein classification;ensemble
                                        classifier;bootstrapping;Sat-image
                                        classification;medical diagnoses},
                            doi={10.1109/ICPR.2014.258},
                            ISSN={1051-4651},
                            month={Aug}}

    Notes:
        * the non-linear approach is inefficient
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 method='linear',
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            method (str): 'linear'/'non-linear_2.0' - the float can be any
                                number: standard deviation in the
                                Gaussian-kernel
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        if not method == 'linear' and not method.startswith('non-linear'):
            raise ValueError(self.__class__.__name__ + ": " +
                             'Method parameter %s is not supported' % method)
        elif method.startswith('non-linear'):
            parameter = float(method.split('_')[-1])
            if parameter <= 0:
                message = ("Non-positive non-linear parameter %f is "
                           "not supported") % parameter
                raise ValueError(self.__class__.__name__ + ": " + message)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.method = method
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'method': ['linear', 'non-linear_0.1',
                                             'non-linear_1.0',
                                             'non-linear_2.0']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if self.method == 'linear':
            # finding H_l by linear decomposition
            cov = np.cov(X_min, rowvar=False)
            w, v = np.linalg.eig(cov)
            H_l = v[np.argmax(w)]
        else:
            # building a non-linear kernel matrix and finding H_n by its
            # decomposition
            self.sigma = float(self.method.split('_')[-1])
            kernel_matrix = pairwise_distances(X_min)
            kernel_matrix = kernel_matrix/(2.0*self.sigma**2)
            kernel_matrix = np.exp(kernel_matrix)
            try:
                w_k, v_k = np.linalg.eig(kernel_matrix)
            except Exception as e:
                return X.copy(), y.copy()
            H_n = v_k[np.argmax(w_k)]

            def kernel(x, y):
                return np.linalg.norm(x - y)/(2.0*self.sigma**2)

        # generating samples
        samples = []

        def angle(P, n, H_l):
            numerator = np.abs(np.dot(P[n], H_l))
            denominator = np.linalg.norm(P[n])*np.linalg.norm(H_l)
            return np.arccos(numerator/denominator)

        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            # calculating difference vectors from all neighbors
            P = X_min[ind[idx][1:]] - X_min[idx]
            if self.method == 'linear':
                # calculating angles with the principal direction
                thetas = np.array([angle(P, n, H_l) for n in range(len(P))])
            else:
                thetas = []
                # calculating angles of the difference vectors and the
                # principal direction in feature space
                for n in range(len(P)):
                    # calculating representation in feature space
                    feature_vector = np.array(
                        [kernel(X_min[k], P[n]) for k in range(len(X_min))])
                    dp = np.dot(H_n, feature_vector)
                    denom = np.linalg.norm(feature_vector)*np.linalg.norm(H_n)
                    thetas.append(np.arccos(np.abs(dp)/denom))
                thetas = np.array(thetas)

            # using the neighbor with the difference along the most similar
            # direction to the principal direction of the data
            n = np.argmin(thetas)
            X_a = X_min[idx]
            X_b = X_min[ind[idx][1:][n]]
            samples.append(self.sample_between_points_componentwise(X_a, X_b))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'method': self.method,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class NT_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{nt_smote,
                            author={Xu, Y. H. and Li, H. and Le, L. P. and
                                        Tian, X. Y.},
                            booktitle={2014 Seventh International Joint
                                        Conference on Computational Sciences
                                        and Optimization},
                            title={Neighborhood Triangular Synthetic Minority
                                    Over-sampling Technique for Imbalanced
                                    Prediction on Small Samples of Chinese
                                    Tourism and Hospitality Firms},
                            year={2014},
                            volume={},
                            number={},
                            pages={534-538},
                            keywords={financial management;pattern
                                        classification;risk management;sampling
                                        methods;travel industry;Chinese
                                        tourism; hospitality firms;imbalanced
                                        risk prediction;minority class samples;
                                        up-sampling approach;neighborhood
                                        triangular synthetic minority
                                        over-sampling technique;NT-SMOTE;
                                        nearest neighbor idea;triangular area
                                        sampling idea;single classifiers;data
                                        excavation principles;hospitality
                                        industry;missing financial indicators;
                                        financial data filtering;financial risk
                                        prediction;MDA;DT;LSVM;logit;probit;
                                        firm risk prediction;Joints;
                                        Optimization;imbalanced datasets;
                                        NT-SMOTE;neighborhood triangular;
                                        random sampling},
                            doi={10.1109/CSO.2014.104},
                            ISSN={},
                            month={July}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_application]

    def __init__(self, proportion=1.0, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # find two nearest minority samples
        nn = NearestNeighbors(n_neighbors=3, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        samples = []
        while len(samples) < n_to_sample:
            # select point randomly
            idx = self.random_state.randint(len(X_min))
            P_1 = X_min[idx]
            # find two closest neighbors
            P_2 = X_min[ind[idx][1]]
            P_3 = X_min[ind[idx][2]]
            # generate random point by sampling the specified triangle
            r_1 = self.random_state.random_sample()
            r_2 = self.random_state.random_sample()
            samples.append((P_3 + r_1 * ((P_1 + r_2 * (P_2 - P_1)) - P_3)))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Lee(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{lee,
                             author = {Lee, Jaedong and Kim,
                                 Noo-ri and Lee, Jee-Hyong},
                             title = {An Over-sampling Technique with Rejection
                                        for Imbalanced Class Learning},
                             booktitle = {Proceedings of the 9th International
                                            Conference on Ubiquitous
                                            Information Management and
                                            Communication},
                             series = {IMCOM '15},
                             year = {2015},
                             isbn = {978-1-4503-3377-1},
                             location = {Bali, Indonesia},
                             pages = {102:1--102:6},
                             articleno = {102},
                             numpages = {6},
                             doi = {10.1145/2701126.2701181},
                             acmid = {2701181},
                             publisher = {ACM},
                             address = {New York, NY, USA},
                             keywords = {data distribution, data preprocessing,
                                            imbalanced problem, rejection rule,
                                            synthetic minority oversampling
                                            technique}
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 rejection_level=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbor
                                component
            rejection_level (float): the rejection level of generated samples,
                                        if the fraction of majority labels in
                                        the local environment is higher than
                                        this number, the generated point is
                                        rejected
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(rejection_level, "rejection_level", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.rejection_level = rejection_level
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'rejection_level': [0.3, 0.5, 0.7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors models to find neighbors of minority
        # samples in the total data and in the minority datasets
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn_min = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn_min.fit(X_min)
        dist_min, ind_min = nn_min.kneighbors(X_min)

        # do the sampling, we impleneted a continouos tweaking of rejection
        # levels in order to fix situations when no unrejectable data can
        # be can be generated
        samples = []
        passed = 0
        trial = 0
        rejection_level = self.rejection_level
        while len(samples) < n_to_sample:
            # checking if we managed to generate a single data in 1000 trials
            if passed == trial and passed > 1000:
                rejection_level = rejection_level + 0.1
                trial = 0
                passed = 0
            trial = trial + 1
            # generating random point
            idx = self.random_state.randint(len(X_min))
            random_neighbor_idx = self.random_state.choice(ind_min[idx][1:])
            X_a = X_min[idx]
            X_b = X_min[random_neighbor_idx]
            random_point = self.sample_between_points(X_a, X_b)
            # checking if the local environment is above the rejection level
            dist_new, ind_new = nn.kneighbors(random_point.reshape(1, -1))
            maj_frac = np.sum(y[ind_new][:-1] ==
                              self.maj_label)/self.n_neighbors
            if maj_frac < rejection_level:
                samples.append(random_point)
            else:
                passed = passed + 1

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'rejection_level': self.rejection_level,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SPY(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{spy,
                            author={Dang, X. T. and Tran, D. H. and Hirose, O.
                                    and Satou, K.},
                            booktitle={2015 Seventh International Conference
                                        on Knowledge and Systems Engineering
                                        (KSE)},
                            title={SPY: A Novel Resampling Method for
                                    Improving Classification Performance in
                                    Imbalanced Data},
                            year={2015},
                            volume={},
                            number={},
                            pages={280-285},
                            keywords={decision making;learning (artificial
                                        intelligence);pattern classification;
                                        sampling methods;SPY;resampling
                                        method;decision-making process;
                                        biomedical data classification;
                                        class imbalance learning method;
                                        SMOTE;oversampling method;UCI
                                        machine learning repository;G-mean
                                        value;borderline-SMOTE;
                                        safe-level-SMOTE;Support vector
                                        machines;Training;Bioinformatics;
                                        Proteins;Protein engineering;Radio
                                        frequency;Sensitivity;Imbalanced
                                        dataset;Over-sampling;
                                        Under-sampling;SMOTE;
                                        borderline-SMOTE},
                            doi={10.1109/KSE.2015.24},
                            ISSN={},
                            month={Oct}}
    """

    categories = [OverSampling.cat_changes_majority]

    def __init__(self,
                 n_neighbors=5,
                 threshold=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors in nearest neighbor
                                component
            threshold (float): threshold*n_neighbors gives the threshold z
                                described in the paper
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(threshold, "threshold", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.n_jobs = n_jobs

        # random state takes no effect for this technique
        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'threshold': [0.3, 0.5, 0.7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        y_new = y.copy()
        z = self.threshold*n_neighbors

        # checking the neighbors of each minority sample
        for i in range(len(X_min)):
            majority_mask = y[ind[i][1:]] == self.maj_label
            x = np.sum(majority_mask)
            # if the number of majority samples in the neighborhood is
            # smaller than a threshold
            # their labels are changed to minority
            if x < z:
                y_new[ind[i][1:][majority_mask]] = self.min_label

        return X.copy(), y_new

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'threshold': self.threshold,
                'n_jobs': self.n_jobs}


class SMOTE_PSOBAT(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{smote_psobat,
                            author={Li, J. and Fong, S. and Zhuang, Y.},
                            booktitle={2015 3rd International Symposium on
                                        Computational and Business
                                        Intelligence (ISCBI)},
                            title={Optimizing SMOTE by Metaheuristics with
                                    Neural Network and Decision Tree},
                            year={2015},
                            volume={},
                            number={},
                            pages={26-32},
                            keywords={data mining;particle swarm
                                        optimisation;pattern classification;
                                        data mining;classifier;metaherustics;
                                        SMOTE parameters;performance
                                        indicators;selection optimization;
                                        PSO;particle swarm optimization
                                        algorithm;BAT;bat-inspired algorithm;
                                        metaheuristic optimization algorithms;
                                        nearest neighbors;imbalanced dataset
                                        problem;synthetic minority
                                        over-sampling technique;decision tree;
                                        neural network;Classification
                                        algorithms;Neural networks;Decision
                                        trees;Training;Optimization;Particle
                                        swarm optimization;Data mining;SMOTE;
                                        Swarm Intelligence;parameter
                                        selection optimization},
                            doi={10.1109/ISCBI.2015.12},
                            ISSN={},
                            month={Dec}}

    Notes:
        * The parameters of the memetic algorithms are not specified.
        * I have checked multiple paper describing the BAT algorithm, but the
            meaning of "Generate a new solution by flying randomly" is still
            unclear.
        * It is also unclear if best solutions are recorded for each bat, or
            the entire population.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_memetic]

    def __init__(self,
                 maxit=50,
                 c1=0.3,
                 c2=0.1,
                 c3=0.1,
                 alpha=0.9,
                 gamma=0.9,
                 method='bat',
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            maxit (int): maximum number of iterations
            c1 (float): intertia weight of PSO
            c2 (float): attraction of local maximums in PSO
            c3 (float): attraction of global maximum in PSO
            alpha (float): alpha parameter of the method
            gamma (float): gamma parameter of the method
            method (str): optimization technique to be used
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(maxit, "maxit", 1)
        self.check_greater_or_equal(c1, "c1", 0)
        self.check_greater_or_equal(c2, "c2", 0)
        self.check_greater_or_equal(c3, "c3", 0)
        self.check_greater_or_equal(alpha, "alpha", 0)
        self.check_greater_or_equal(gamma, "gamma", 0)
        self.check_isin(method, "method", ['pso', 'bat'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.maxit = maxit
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.alpha = alpha
        self.gamma = gamma
        self.method = method
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        bat_pc = cls.generate_parameter_combinations({'maxit': [50],
                                                      'alpha': [0.7, 0.9],
                                                      'gamma': [0.7, 0.9],
                                                      'method': ['bat']}, raw)
        pso_pc = cls.generate_parameter_combinations({'maxit': [50],
                                                      'c1': [0.2, 0.5],
                                                      'c2': [0.1, 0.2],
                                                      'c3': [0.1, 0.2],
                                                      'method': ['pso']}, raw)
        if not raw:
            bat_pc.extend(pso_pc)
        else:
            bat_pc = {**bat_pc, **pso_pc}
        return bat_pc

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        def evaluate(K, proportion):
            """
            Evaluate given configuration

            Args:
                K (int): number of neighbors in nearest neighbors component
                proportion (float): proportion of missing data to generate

            Returns:
                float, float: kappa and accuracy scores
            """
            smote = SMOTE(proportion=proportion,
                          n_neighbors=K,
                          n_jobs=self.n_jobs,
                          random_state=self.random_state)
            X_samp, y_samp = smote.sample(X, y)

            # doing k-fold cross validation
            kfold = KFold(5)
            preds = []
            tests = []
            for train, test in kfold.split(X_samp):
                dt = DecisionTreeClassifier(random_state=self.random_state)
                dt.fit(X_samp[train], y_samp[train])
                preds.append(dt.predict(X_samp[test]))
                tests.append(y_samp[test])
            preds = np.hstack(preds)
            tests = np.hstack(tests)
            # computing the kappa score
            tp = np.sum(np.logical_and(preds == tests,
                                       tests == self.min_label))
            fn = np.sum(np.logical_and(preds != tests,
                                       tests == self.min_label))
            tn = np.sum(np.logical_and(preds == tests,
                                       tests == self.maj_label))
            fp = np.sum(np.logical_and(preds != tests,
                                       tests == self.maj_label))

            p_o = (tp + tn)/(tp + fn + tn + fp)
            p_e = (tp + fn)*(tp + fp)/(tp + fn + tn + fp)**2 + \
                (fp + tn)*(fn + tn)/(tp + fn + tn + fp)**2

            kappa = (p_o - p_e)/(1.0 - p_e)

            return kappa, p_o

        def PSO():
            """
            PSO optimization

            Returns:
                int, float: the best K and proportion values
            """
            # a reasonable range of nearest neighbors to use with SMOTE
            k_range = [2, min([np.sum(y == self.min_label), 10])]
            # a reasonable range of proportions
            proportion_range = [0.1, 2.0]
            # population size
            n_pop = 10

            # initial particles
            def init_particle():
                k_rand = self.random_state.randint(k_range[0], k_range[1])
                r = self.random_state.random_sample()
                diff = proportion_range[1] - proportion_range[0]
                vect = r*diff + proportion_range[0]
                return np.array([k_rand, vect])
            ps = [init_particle() for _ in range(n_pop)]
            # initial velocities
            velocities = [np.array([0, 0]) for _ in range(n_pop)]
            # best configurations of particles
            local_best = [ps[i].copy() for i in range(n_pop)]
            # scores of best configurations of particles
            local_scores = [(0, 0) for _ in range(n_pop)]
            # global best configuration of particles
            global_best = ps[0].copy()
            # global best score
            global_scores = (0, 0)

            # executing the particle swarm optimization
            not_changed = 0
            for _ in range(self.maxit):
                # if the configurations didn't change for 10 iterations, stop
                if not_changed > len(ps)*10:
                    break
                # evaluating each of the configurations
                for i in range(len(ps)):
                    scores = evaluate(np.int(ps[i][0]), ps[i][1])
                    # recording if the best scores didn't change
                    not_changed = not_changed + 1
                    # registering locally and globally best scores
                    if (min([local_scores[i][0], scores[0]]) > 0.4
                            and local_scores[i][1] > scores[1]):
                        local_scores[i] = scores
                        local_best[i] = ps[i].copy()
                        not_changed = 0
                    elif scores[0] > 0.4 and local_scores[i][0] <= 0.4:
                        local_scores[i] = scores
                        local_best[i] = ps[i].copy()
                        not_changed = 0

                    if (min([global_scores[0], scores[0]]) > 0.4
                            and global_scores[1] > scores[1]):
                        global_scores = scores
                        global_best = ps[i].copy()
                        not_changed = 0
                    elif scores[0] > 0.4 and global_scores[0] <= 0.4:
                        global_scores = scores
                        global_best = ps[i].copy()
                        not_changed = 0

                # update velocities
                for i in range(len(ps)):
                    velocities[i] = self.c1*velocities[i] + \
                        (local_best[i] - ps[i])*self.c2 + \
                        (global_best - ps[i])*self.c3
                    # clipping velocities if required
                    while abs(velocities[i][0]) > k_range[1] - k_range[0]:
                        velocities[i][0] = velocities[i][0]/2.0
                    diff = proportion_range[1] - proportion_range[0]
                    while abs(velocities[i][1]) > diff:
                        velocities[i][1] = velocities[i][1]/2.0

                # update positions
                for i in range(len(ps)):
                    ps[i] = ps[i] + velocities[i]
                    # clipping positions according to the specified ranges
                    ps[i][0] = np.clip(ps[i][0], k_range[0], k_range[1])
                    ps[i][1] = np.clip(ps[i][1],
                                       proportion_range[0],
                                       proportion_range[1])

            return global_best

        def BAT():
            """
            BAT optimization

            Returns:
                int, float: the best K and proportion values
            """

            if sum(y == self.min_label) < 2:
                return X.copy(), y.copy()

            # a reasonable range of nearest neighbors to use with SMOTE
            k_range = [1, min([np.sum(y == self.min_label), 10])]
            # a reasonable range of proportions
            proportion_range = [0.1, 2.0]
            # population size
            n_pop = 10
            # maximum frequency
            f_max = 10

            def init_bat():
                k_rand = self.random_state.randint(k_range[0], k_range[1])
                r = self.random_state.random_sample()
                diff = proportion_range[1] - proportion_range[0]
                return np.array([k_rand, r*diff + proportion_range[0]])

            # initial bat positions
            bats = [init_bat() for _ in range(n_pop)]
            # initial velocities
            velocities = [np.array([0, 0]) for _ in range(10)]
            # best configurations of particles
            local_best = [[[[0.0, 0.0], bats[i].copy()]]
                          for i in range(len(bats))]
            # scores of best configurations of particles
            global_best = [[0.0, 0.0], bats[0].copy()]
            # pulse frequencies
            f = self.random_state.random_sample(size=n_pop)*f_max
            # pulse rates
            r = self.random_state.random_sample(size=n_pop)
            # loudness
            A = self.random_state.random_sample(size=n_pop)

            # gamma parameter according to the BAT paper
            gamma = self.gamma
            # alpha parameter according to the BAT paper
            alpha = self.alpha

            # initial best solution
            bat_star = bats[0].copy()

            not_changed = 0
            for t in range(self.maxit):
                not_changed = not_changed + 1

                if not_changed > 10:
                    break

                # update frequencies
                f = self.random_state.random_sample(size=n_pop)*f_max

                # update velocities
                for i in range(len(velocities)):
                    velocities[i] = velocities[i] + (bats[i] - bat_star)*f[i]

                # update bats
                for i in range(len(bats)):
                    bats[i] = bats[i] + velocities[i]
                    bats[i][0] = np.clip(bats[i][0], k_range[0], k_range[1])
                    bats[i][1] = np.clip(
                        bats[i][1], proportion_range[0], proportion_range[1])

                for i in range(n_pop):
                    # generate local solution
                    if self.random_state.random_sample() > r[i]:
                        n_rand = min([len(local_best[i]), 5])
                        rand_int = self.random_state.randint(n_rand)
                        random_best_sol = local_best[i][rand_int][1]
                        rr = self.random_state.random_sample(
                            size=len(bat_star))
                        bats[i] = random_best_sol + rr*A[i]

                # evaluate and do local search
                for i in range(n_pop):
                    scores = evaluate(int(bats[i][0]), bats[i][1])

                    # checking if the scores are better than the global score
                    # implementation of the multi-objective criterion in the
                    # SMOTE-PSOBAT paper
                    improved_global = False
                    if (min([global_best[0][0], scores[0]]) > 0.4
                            and global_best[0][1] > scores[1]):
                        improved_global = True
                        not_changed = 0
                    elif scores[0] > 0.4 and global_best[0][0] <= 0.4:
                        improved_global = True
                        not_changed = 0

                    # checking if the scores are better than the local scores
                    # implementation of the multi-objective criterion in the
                    # SMOTE-PSOBAT paper
                    improved_local = False
                    if (min([local_best[i][0][0][0], scores[0]]) > 0.4
                            and local_best[i][0][0][1] > scores[1]):
                        improved_local = True
                    elif scores[0] > 0.4 and local_best[i][0][0][0] <= 0.4:
                        improved_local = True

                    # local search in the bet algorithm
                    if (self.random_state.random_sample() < A[i]
                            and improved_local):
                        local_best[i].append([scores, bats[i].copy()])
                        A[i] = A[i]*alpha
                        r[i] = r[i]*(1 - np.exp(-gamma*t))
                    if (self.random_state.random_sample() < A[i]
                            and improved_global):
                        global_best = [scores, bats[i].copy()]

                    # ranking local solutions to keep track of the best 5
                    local_best[i] = sorted(
                        local_best[i], key=lambda x: -x[0][0])
                    local_best[i] = local_best[i][:min(
                        [len(local_best[i]), 5])]

                t = t + 1

            return global_best[1]

        if self.method == 'pso':
            best_combination = PSO()
        elif self.method == 'bat':
            best_combination = BAT()
        else:
            message = "Search method %s not supported yet." % self.method
            raise ValueError(self.__class__.__name__ + ": " + message)

        return SMOTE(proportion=best_combination[1],
                     n_neighbors=int(best_combination[0]),
                     n_jobs=self.n_jobs,
                     random_state=self.random_state).sample(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'maxit': self.maxit,
                'c1': self.c1,
                'c2': self.c2,
                'c3': self.c3,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'method': self.method,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class MDO(OverSampling):
    """
    References:
        * BibTex::

            @ARTICLE{mdo,
                        author={Abdi, L. and Hashemi, S.},
                        journal={IEEE Transactions on Knowledge and Data
                                    Engineering},
                        title={To Combat Multi-Class Imbalanced Problems
                                by Means of Over-Sampling Techniques},
                        year={2016},
                        volume={28},
                        number={1},
                        pages={238-251},
                        keywords={covariance analysis;learning (artificial
                                    intelligence);modelling;pattern
                                    classification;sampling methods;
                                    statistical distributions;minority
                                    class instance modelling;probability
                                    contour;covariance structure;MDO;
                                    Mahalanobis distance-based oversampling
                                    technique;data-oriented technique;
                                    model-oriented solution;machine learning
                                    algorithm;data skewness;multiclass
                                    imbalanced problem;Mathematical model;
                                    Training;Accuracy;Eigenvalues and
                                    eigenfunctions;Machine learning
                                    algorithms;Algorithm design and analysis;
                                    Benchmark testing;Multi-class imbalance
                                    problems;over-sampling techniques;
                                    Mahalanobis distance;Multi-class imbalance
                                    problems;over-sampling techniques;
                                    Mahalanobis distance},
                        doi={10.1109/TKDE.2015.2458858},
                        ISSN={1041-4347},
                        month={Jan}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction]

    def __init__(self,
                 proportion=1.0,
                 K2=5,
                 K1_frac=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            K2 (int): number of neighbors
            K1_frac (float): the fraction of K2 to set K1
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(K2, "K2", 1)
        self.check_greater_or_equal(K1_frac, "K1_frac", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.K2 = K2
        self.K1_frac = K1_frac
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'K2': [3, 5, 7],
                                  'K1_frac': [0.3, 0.5, 0.7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # determining K1
        self.K1 = int(self.K2*self.K1_frac)
        K1 = min([self.K1, len(X)])
        K2 = min([self.K2 + 1, len(X)])

        # Algorithm 2 - chooseSamples
        nn = NearestNeighbors(n_neighbors=K2, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        # extracting the number of minority samples in local neighborhoods
        n_min = np.array([np.sum(y[ind[i][1:]] == self.min_label)
                          for i in range(len(X_min))])

        # extracting selected samples from minority ones
        X_sel = X_min[n_min >= K1]

        # falling back to returning input data if all the input is considered
        # noise
        if len(X_sel) == 0:
            _logger.info(self.__class__.__name__ +
                         ": " + "No samples selected")
            return X.copy(), y.copy()

        # computing distribution
        weights = n_min[n_min >= K1]/K2
        weights = weights/np.sum(weights)

        # Algorithm 1 - MDO over-sampling
        mu = np.mean(X_sel, axis=0)
        Z = X_sel - mu
        # executing PCA
        pca = PCA(n_components=min([len(Z[0]), len(Z)])).fit(Z)
        T = pca.transform(Z)
        # computing variances (step 13)
        V = np.var(T, axis=0)

        V[V < 0.001] = 0.001

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            # selecting a sample randomly according to the distribution
            idx = self.random_state.choice(np.arange(len(X_sel)), p=weights)

            # finding vector in PCA space
            X_temp = T[idx]
            X_temp_square = X_temp**2

            # computing alphas
            alpha = np.sum(X_temp_square/V)
            alpha_V = alpha*V
            alpha_V[alpha_V < 0.001] = 0.001

            # initializing a new vector
            X_new = np.zeros(len(X_temp))

            # sampling components of the new vector
            s = 0
            for j in range(len(X_temp)-1):
                r = (2*self.random_state.random_sample()-1)*np.sqrt(alpha_V[j])
                X_new[j] = r
                s = s + (r**2/alpha_V[j])

            if s > 1:
                last_fea_val = 0
            else:
                tmp = (1 - s)*alpha*V[-1]
                if tmp < 0:
                    tmp = 0
                last_fea_val = np.sqrt(tmp)
            # determine last component to fulfill the ellipse equation
            X_new[-1] = (2*self.random_state.random_sample()-1)*last_fea_val
            # append to new samples
            samples.append(X_new)

        return (np.vstack([X, pca.inverse_transform(samples) + mu]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'K2': self.K2,
                'K1_frac': self.K1_frac,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Random_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{random_smote,
                            author="Dong, Yanjie
                            and Wang, Xuehua",
                            editor="Xiong, Hui
                            and Lee, W. B.",
                            title="A New Over-Sampling Approach: Random-SMOTE
                                    for Learning from Imbalanced Data Sets",
                            booktitle="Knowledge Science, Engineering and
                                        Management",
                            year="2011",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="343--352",
                            isbn="978-3-642-25975-3"
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): number of neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn

        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model to find closest neighbors of minority
        # points
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)))
            y_1_idx, y_2_idx = self.random_state.choice(ind[idx][1:], 2)
            t = self.sample_between_points_componentwise(
                X_min[y_1_idx], X_min[y_2_idx])
            samples.append(
                self.sample_between_points_componentwise(X_min[idx], t))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ISMOTE(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{ismote,
                            author="Li, Hu
                            and Zou, Peng
                            and Wang, Xiang
                            and Xia, Rongze",
                            editor="Sun, Zengqi
                            and Deng, Zhidong",
                            title="A New Combination Sampling Method for
                                    Imbalanced Data",
                            booktitle="Proceedings of 2013 Chinese Intelligent
                                        Automation Conference",
                            year="2013",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="547--554",
                            isbn="978-3-642-38466-0"
                            }
    """

    categories = [OverSampling.cat_changes_majority]

    def __init__(self,
                 n_neighbors=5,
                 minority_weight=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors
            minority_weight (float): weight parameter according to the paper
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(minority_weight, "minority_weight", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.minority_weight = minority_weight
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'minority_weight': [0.2, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_to_sample = int((len(X_maj) - len(X_min))/2 + 0.5)

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # computing distances of majority samples from minority ones
        nn = NearestNeighbors(n_neighbors=len(X_min), n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_maj)

        # sort majority instances in descending order by their mean distance
        # from minority samples
        to_sort = zip(np.arange(len(X_maj)), np.mean(dist, axis=1))
        ind_sorted, dist_sorted = zip(*sorted(to_sort, key=lambda x: -x[1]))

        # remove the ones being farthest from the minority samples
        X_maj = X_maj[list(ind_sorted[n_to_sample:])]

        # construct new dataset
        X_new = np.vstack([X_maj, X_min])
        y_new = np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min))])

        X_min = X_new[y_new == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_new), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_new)
        dist, ind = nn.kneighbors(X_min)

        # do the oversampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)))
            y_idx = self.random_state.choice(ind[idx][1:])

            # different generation scheme depending on the class label
            if y_new[y_idx] == self.min_label:
                diff = (X_new[y_idx] - X_min[idx])
                r = self.random_state.random_sample()
                samples.append(X_min[idx] + r * diff * self.minority_weight)
            else:
                diff = (X_new[y_idx] - X_min[idx])
                r = self.random_state.random_sample()
                sample = X_min[idx] + r * diff * (1.0 - self.minority_weight)
                samples.append(sample)

        return (np.vstack([X_new, np.vstack(samples)]),
                np.hstack([y_new, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'minority_weight': self.minority_weight,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class VIS_RST(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{vis_rst,
                            author="Borowska, Katarzyna
                            and Stepaniuk, Jaroslaw",
                            editor="Saeed, Khalid
                            and Homenda, Wladyslaw",
                            title="Imbalanced Data Classification: A Novel
                                    Re-sampling Approach Combining Versatile
                                    Improved SMOTE and Rough Sets",
                            booktitle="Computer Information Systems and
                                        Industrial Management",
                            year="2016",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="31--42",
                            isbn="978-3-319-45378-1"
                            }

    Notes:
        * Replication of DANGER samples will be removed by the last step of
            noise filtering.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_noise_removal]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardizing the data
        ss = StandardScaler()
        ss.fit(X)
        X = ss.transform(X)
        y = y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model to determine boundary region
        n_neighbors = min([len(X), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_maj)

        # determining boundary region of majority samples
        boundary = np.array([np.sum(y[ind[i]] == self.maj_label)
                             != n_neighbors for i in range(len(X_maj))])
        y_maj = y[y == self.maj_label]
        y_maj[boundary] = self.min_label
        y[y == self.maj_label] = y_maj

        # extracting new minority and majority set
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # labeling minority samples
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        # extracting labels
        labels = []
        for i in range(len(ind)):
            min_class_neighbors = np.sum(y[ind[i][1:]] == self.maj_label)
            if min_class_neighbors == n_neighbors-1:
                labels.append('noise')
            elif min_class_neighbors < n_neighbors/2:
                labels.append('safe')
            else:
                labels.append('danger')

        # extracting the number of different labels (noise is not used)
        safe = np.sum([li == 'safe' for li in labels])
        danger = np.sum([li == 'danger' for li in labels])

        if safe == 0:
            mode = 'no_safe'
        elif danger > 0.3*len(X_min):
            mode = 'high_complexity'
        else:
            mode = 'low_complexity'

        # fitting nearest neighbors to find the neighbors of minority elements
        # among minority elements
        n_neighbors_min = min([len(X_min), self.n_neighbors + 1])
        nn_min = NearestNeighbors(n_neighbors=n_neighbors_min,
                                  n_jobs=self.n_jobs)
        nn_min.fit(X_min)
        dist_min, ind_min = nn_min.kneighbors(X_min)

        # do the sampling
        samples = []
        mask = np.repeat(False, len(X_min))
        while len(samples) < n_to_sample:
            # choosing a random minority sample
            idx = self.random_state.choice(np.arange(len(X_min)))

            # implementation of sampling rules depending on the mode
            if mode == 'high_complexity':
                if labels[idx] == 'noise':
                    pass
                elif labels[idx] == 'danger' and not mask[idx]:
                    samples.append(X_min[idx])
                    mask[idx] = True
                else:
                    X_b = X_min[self.random_state.choice(ind_min[idx][1:])]
                    samples.append(self.sample_between_points(X_min[idx], X_b))
            elif mode == 'low_complexity':
                if labels[idx] == 'noise':
                    pass
                elif labels[idx] == 'danger':
                    X_b = X_min[self.random_state.choice(ind_min[idx][1:])]
                    samples.append(self.sample_between_points(X_min[idx], X_b))
                elif not mask[idx]:
                    samples.append(X_min[idx])
                    mask[idx] = True
            else:
                X_b = X_min[self.random_state.choice(ind_min[idx][1:])]
                samples.add(self.sample_between_points(X_min[idx], X_b))

        X_samp = np.vstack(samples)

        # final noise removal by removing those minority samples generated
        # and not belonging to the lower approximation
        nn = NearestNeighbors(n_neighbors=n_neighbors,
                              n_jobs=self.n_jobs).fit(X)
        dist_check, ind_check = nn.kneighbors(X_samp)

        def maj_zero(i):
            return np.sum(y[ind_check[i][1:]] == self.maj_label) == 0

        num_maj_mask = np.array([maj_zero(i) for i in range(len(samples))])
        X_samp = X_samp[num_maj_mask]

        return (ss.inverse_transform(np.vstack([X, X_samp])),
                np.hstack([y, np.repeat(self.min_label, len(X_samp))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class GASMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{gasmote,
                        author="Jiang, Kun
                        and Lu, Jing
                        and Xia, Kuiliang",
                        title="A Novel Algorithm for Imbalance Data
                                Classification Based on Genetic
                                Algorithm Improved SMOTE",
                        journal="Arabian Journal for Science and
                                    Engineering",
                        year="2016",
                        month="Aug",
                        day="01",
                        volume="41",
                        number="8",
                        pages="3255--3266",
                        issn="2191-4281",
                        doi="10.1007/s13369-016-2179-2",
                        url="https://doi.org/10.1007/s13369-016-2179-2"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_memetic,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 n_neighbors=5,
                 maxn=7,
                 n_pop=10,
                 popl3=5,
                 pm=0.3,
                 pr=0.2,
                 Ge=10,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors
            maxn (int): maximum number of samples to generate per minority
                        instances
            n_pop (int): size of population
            popl3 (int): number of crossovers
            pm (float): mutation probability
            pr (float): selection probability
            Ge (int): number of generations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(maxn, "maxn", 1)
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_in_range(pm, "pm", [0, 1])
        self.check_in_range(pr, "pr", [0, 1])
        self.check_greater_or_equal(Ge, "Ge", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.maxn = maxn
        self.n_pop = n_pop
        self.popl3 = popl3
        self.pm = pm
        self.pr = pr
        self.Ge = Ge
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        return cls.generate_parameter_combinations({'n_neighbors': [7],
                                                    'maxn': [2, 3, 4],
                                                    'n_pop': [10],
                                                    'popl3': [4],
                                                    'pm': [0.3],
                                                    'pr': [0.2],
                                                    'Ge': [10]}, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model to find minority neighbors of
        #  minority samples
        n_neighbors = min([self.n_neighbors + 1, len(X_min)])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)
        kfold = KFold(min([len(X), 5]))

        def fitness(conf):
            """
            Evluate fitness of configuration

            Args:
                conf (list(list)): configuration
            """
            # generate new samples
            samples = []
            for i in range(len(conf)):
                for _ in range(conf[i]):
                    X_b = X_min[self.random_state.choice(ind[i][1:])]
                    samples.append(self.sample_between_points(X_min[i], X_b))

            if len(samples) == 0:
                # if no samples are generated
                X_new = X
                y_new = y
            else:
                # construct dataset
                X_new = np.vstack([X, np.vstack(samples)])
                y_new = np.hstack(
                    [y, np.repeat(self.min_label, len(samples))])

            # execute kfold cross validation
            preds, tests = [], []
            for train, test in kfold.split(X_new):
                dt = DecisionTreeClassifier(random_state=self.random_state)
                dt.fit(X_new[train], y_new[train])
                preds.append(dt.predict(X_new[test]))
                tests.append(y_new[test])
            preds = np.hstack(preds)
            tests = np.hstack(tests)

            # compute fitness measure
            tp = np.sum(np.logical_and(
                tests == self.min_label, tests == preds))
            tn = np.sum(np.logical_and(
                tests == self.maj_label, tests == preds))
            fp = np.sum(np.logical_and(
                tests == self.maj_label, tests != preds))
            fn = np.sum(np.logical_and(
                tests == self.min_label, tests != preds))
            sens = tp/(tp + fn)
            spec = tn/(fp + tn)

            return np.sqrt(sens*spec)

        def crossover(conf_a, conf_b):
            """
            Crossover

            Args:
                conf_a (list(list)): configuration to crossover
                conf_b (list(list)): configuration to crossover

            Returns:
                list(list), list(list): the configurations after crossover
            """
            for _ in range(self.popl3):
                k = self.random_state.randint(len(conf_a))
                conf_a = np.hstack([conf_a[:k], conf_b[k:]])
                conf_b = np.hstack([conf_b[:k], conf_a[k:]])
            return conf_a, conf_b

        def mutation(conf, ge):
            """
            Mutation

            Args:
                conf (list(list)): configuration to mutate
                ge (int): iteration number
            """
            conf = conf.copy()
            if self.random_state.random_sample() < self.pm:
                pass
            else:
                for i in range(len(conf)):
                    r = self.random_state.random_sample()
                    r = r**((1 - ge/self.Ge)**3)
                    if self.random_state.randint(2) == 0:
                        conf[i] = int(conf[i] + (self.maxn - conf[i])*r)
                    else:
                        conf[i] = int(conf[i] - (conf[i] - 0)*r)
            return conf

        # generate initial population
        def init_pop():
            return self.random_state.randint(self.maxn, size=len(X_min))

        population = [[init_pop(), 0] for _ in range(self.n_pop)]

        # calculate fitness values
        for p in population:
            p[1] = fitness(p[0])

        # start iteration
        ge = 0
        while ge < self.Ge:
            # sorting population in descending order by fitness scores
            population = sorted(population, key=lambda x: -x[1])

            # selection operation (Step 2)
            pp = int(self.n_pop*self.pr)
            population_new = []
            for i in range(pp):
                population_new.append(population[i])
            population_new.extend(population[:(self.n_pop - pp)])
            population = population_new

            # crossover
            for _ in range(int(self.n_pop/2)):
                pop_0 = population[self.random_state.randint(self.n_pop)][0]
                pop_1 = population[self.random_state.randint(self.n_pop)][0]
                conf_a, conf_b = crossover(pop_0, pop_1)
                population.append([conf_a, fitness(conf_a)])
                population.append([conf_b, fitness(conf_b)])

            # mutation
            for _ in range(int(self.n_pop/2)):
                pop_0 = population[self.random_state.randint(self.n_pop)][0]
                conf = mutation(pop_0, ge)
                population.append([conf, fitness(conf)])

            ge = ge + 1

        # sorting final population
        population = sorted(population, key=lambda x: -x[1])

        # get best configuration
        conf = population[0][0]

        # generate final samples
        samples = []
        for i in range(len(conf)):
            for _ in range(conf[i]):
                samples.append(self.sample_between_points(
                    X_min[i], X_min[self.random_state.choice(ind[i][1:])]))

        if len(samples) == 0:
            return X.copy(), y.copy()

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'maxn': self.maxn,
                'n_pop': self.n_pop,
                'popl3': self.popl3,
                'pm': self.pm,
                'pr': self.pr,
                'Ge': self.Ge,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class A_SUWO(OverSampling):
    """
    References:
        * BibTex::

            @article{a_suwo,
                        title = "Adaptive semi-unsupervised weighted
                                    oversampling (A-SUWO) for imbalanced
                                    datasets",
                        journal = "Expert Systems with Applications",
                        volume = "46",
                        pages = "405 - 416",
                        year = "2016",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2015.10.031",
                        author = "Iman Nekooeimehr and Susana K. Lai-Yuen",
                        keywords = "Imbalanced dataset, Classification,
                                        Clustering, Oversampling"
                        }

    Notes:
        * Equation (7) misses a division by R_j.
        * It is not specified how to sample from clusters with 1 instances.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_noise_removal]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_clus_maj=7,
                 c_thres=0.8,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            n_clus_maj (int): number of majority clusters
            c_thres (float): threshold on distances
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clus_maj, "n_clus_maj", 1)
        self.check_greater_or_equal(c_thres, "c_thres", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_clus_maj = n_clus_maj
        self.c_thres = c_thres
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_clus_maj': [5, 7, 9],
                                  'c_thres': [0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_orig, y_orig = X, y

        # fitting nearest neighbors to find neighbors of all samples
        n_neighbors = min([len(X), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X)

        # identifying as noise those samples which do not have neighbors of
        # the same label
        def noise_func(i):
            return np.sum(y[ind[i][1:]] == y[i]) == 0
        noise = np.where(np.array([noise_func(i) for i in range(len(X))]))[0]

        # removing noise
        X = np.delete(X, noise, axis=0)
        y = np.delete(y, noise)

        # extarcting modified minority and majority datasets
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        if len(X_min) == 0:
            _logger.info("All minority samples removed as noise")
            return X_orig.copy(), y_orig.copy()

        n_clus_maj = min([len(X_maj), self.n_clus_maj])

        # clustering majority samples
        ac = AgglomerativeClustering(n_clusters=n_clus_maj)
        ac.fit(X_maj)
        maj_clusters = [np.where(ac.labels_ == i)[0]
                        for i in range(n_clus_maj)]

        if len(maj_clusters) == 0:
            return X_orig.copy(), y_orig.copy()

        # initialize minority clusters
        min_clusters = [np.array([i]) for i in range(len(X_min))]

        # compute minority distance matrix of cluster
        dm_min = pairwise_distances(X_min)
        for i in range(len(dm_min)):
            dm_min[i, i] = np.inf

        # compute distance matrix of minority and majority clusters
        dm_maj = np.zeros(shape=(len(X_min), len(maj_clusters)))
        for i in range(len(X_min)):
            for j in range(len(maj_clusters)):
                pairwd = pairwise_distances(X_min[min_clusters[i]],
                                            X_maj[maj_clusters[j]])
                dm_maj[i, j] = np.min(pairwd)

        # compute threshold
        nn = NearestNeighbors(n_neighbors=len(X_min), n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)
        d_med = np.median(dist, axis=1)
        T = np.mean(d_med)*self.c_thres

        # do the clustering of minority samples
        while True:
            # finding minimum distance between minority clusters
            pi = np.min(dm_min)

            # if the minimum distance is higher than the threshold, stop
            if pi > T:
                break

            # find cluster pair of minimum distance
            min_dist_pair = np.where(dm_min == pi)
            min_i = min_dist_pair[0][0]
            min_j = min_dist_pair[1][0]

            # Step 3 - find majority clusters closer than pi
            A = np.where(np.logical_and(dm_maj[min_i] < pi,
                                        dm_maj[min_j] < pi))[0]

            # Step 4 - checking if there is a majority cluster between the
            # minority ones
            if len(A) > 0:
                dm_min[min_i, min_j] = np.inf
                dm_min[min_j, min_i] = np.inf
            else:
                # Step 5
                # unifying minority clusters
                min_clusters[min_i] = np.hstack([min_clusters[min_i],
                                                 min_clusters[min_j]])
                # removing one of them
                min_clusters = np.delete(min_clusters, min_j)

                # updating the minority distance matrix
                dm_min[min_i] = np.min(np.vstack([dm_min[min_i],
                                                  dm_min[min_j]]), axis=0)
                dm_min[:, min_i] = dm_min[min_i]
                # removing jth row and column (merged in i)
                dm_min = np.delete(dm_min, min_j, axis=0)
                dm_min = np.delete(dm_min, min_j, axis=1)

                # fixing the diagonal elements
                for i in range(len(dm_min)):
                    dm_min[i, i] = np.inf

                # updating the minority-majority distance matrix
                dm_maj[min_i] = np.min(np.vstack([dm_maj[min_i],
                                                  dm_maj[min_j]]), axis=0)
                dm_maj = np.delete(dm_maj, min_j, axis=0)

        # adaptive sub-cluster sizing
        eps = []
        # going through all minority clusters
        for c in min_clusters:
            # checking if cluster size is higher than 1
            if len(c) > 1:
                k = min([len(c), 5])
                kfold = KFold(k, random_state=self.random_state)
                preds = []
                # executing k-fold cross validation with linear discriminant
                # analysis
                X_c = X_min[c]
                for train, test in kfold.split(X_c):
                    X_train = np.vstack([X_maj, X_c[train]])
                    y_train_maj = np.repeat(self.maj_label, len(X_maj))
                    y_train_min = np.repeat(self.min_label, len(X_c[train]))
                    y_train = np.hstack([y_train_maj, y_train_min])
                    ld = LinearDiscriminantAnalysis()
                    ld.fit(X_train, y_train)
                    preds.append(ld.predict(X_c[test]))
                preds = np.hstack(preds)
                # extracting error rate
                eps.append(np.sum(preds == self.maj_label)/len(preds))
            else:
                eps.append(1.0)

        # sampling distribution over clusters
        min_cluster_dist = eps/np.sum(eps)

        # synthetic instance generation - determining within cluster
        # distribution finding majority neighbor distances of minority
        # samples
        nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)
        dist = dist/len(X[0])
        dist = 1.0/dist

        # computing the THs
        THs = []
        for c in min_clusters:
            THs.append(np.mean(dist[c, 0]))

        # determining within cluster distributions
        within_cluster_dist = []
        for i, c in enumerate(min_clusters):
            Gamma = dist[c, 0]
            Gamma[Gamma > THs[i]] = THs[i]
            within_cluster_dist.append(Gamma/np.sum(Gamma))

        # extracting within cluster neighbors
        within_cluster_neighbors = []
        for c in min_clusters:
            n_neighbors = min([len(c), self.n_neighbors])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            nn.fit(X_min[c])
            within_cluster_neighbors.append(nn.kneighbors(X_min[c])[1])

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # choose random cluster index
            cluster_idx = self.random_state.choice(
                np.arange(len(min_clusters)), p=min_cluster_dist)
            if len(min_clusters[cluster_idx]) > 1:
                # if the cluster has at least two elemenets
                domain = np.arange(len(min_clusters[cluster_idx]))
                distribution = within_cluster_dist[cluster_idx]
                sample_idx = self.random_state.choice(domain, p=distribution)

                domain = within_cluster_neighbors[cluster_idx][sample_idx][1:]
                neighbor_idx = self.random_state.choice(domain)
                point = X_min[min_clusters[cluster_idx][sample_idx]]
                neighbor = X_min[min_clusters[cluster_idx][neighbor_idx]]
                samples.append(self.sample_between_points(point, neighbor))
            else:
                samples.append(X_min[min_clusters[cluster_idx][0]])

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_clus_maj': self.n_clus_maj,
                'c_thres': self.c_thres,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SMOTE_FRST_2T(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_frst_2t,
                        title = "Fuzzy-rough imbalanced learning for the
                                    diagnosis of High Voltage Circuit
                                    Breaker maintenance: The SMOTE-FRST-2T
                                    algorithm",
                        journal = "Engineering Applications of Artificial
                        Intelligence",
                        volume = "48",
                        pages = "134 - 139",
                        year = "2016",
                        issn = "0952-1976",
                        doi = "https://doi.org/10.1016/j.engappai.2015.10.009",
                        author = "Ramentol, E. and Gondres, I. and Lajes, S.
                                    and Bello, R. and Caballero,Y. and
                                    Cornelis, C. and Herrera, F.",
                        keywords = "High Voltage Circuit Breaker (HVCB),
                                    Imbalanced learning, Fuzzy rough set
                                    theory, Resampling methods"
                        }

    Notes:
        * Unlucky setting of parameters might result 0 points added, we have
            fixed this by increasing the gamma_S threshold if the number of
            samples accepted is low.
        * Similarly, unlucky setting of parameters might result all majority
            samples turned into minority.
        * In my opinion, in the algorithm presented in the paper the
            relations are incorrect. The authors talk about accepting samples
            having POS score below a threshold, and in the algorithm in
            both places POS >= gamma is used.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_application]

    def __init__(self,
                 n_neighbors=5,
                 gamma_S=0.7,
                 gamma_M=0.03,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors in the SMOTE sampling
            gamma_S (float): threshold of synthesized samples
            gamma_M (float): threshold of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(gamma_S, "gamma_S", 0)
        self.check_greater_or_equal(gamma_M, "gamma_M", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.gamma_S = gamma_S
        self.gamma_M = gamma_M
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'gamma_S': [0.8, 1.0],
                                  'gamma_M': [0.03, 0.05, 0.1]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # Turning the ranges to 1 speeds up the positive membership
        # calculations
        mmscaler = MinMaxScaler()
        X = mmscaler.fit_transform(X)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # extracting the attribute ranges

        d = len(X[0])

        # after MinMax scaling, the POS value can be calculated as follows
        pos_cache = pairwise_distances(X_min, X_maj, metric='l1')
        pos_cache = 1.0 - pos_cache
        pos_cache = pos_cache.clip(0, d)
        pos_cache = 1.0 - pos_cache

        # initializing some lists containing the results
        result_synth = []
        result_maj = []
        iteration = 0

        gamma_S = self.gamma_S
        gamma_M = self.gamma_M

        # iterating until the dataset becomes balanced
        while (len(X_min) + len(result_synth) + len(result_maj)) < len(X_maj):
            _logger.info(self.__class__.__name__ + ":" +
                         ("iteration: %d" % iteration))
            # checking if the parameters aren't too conservative
            if len(result_synth) < iteration:
                gamma_S = gamma_S*1.1
                _logger.info(self.__class__.__name__ + ": " +
                             "gamma_S increased to %f" % gamma_S)

            # determine proportion
            diff = (sum(y == self.maj_label) -
                    sum(y == self.min_label))
            prop = max(1.1/diff, 0.2)

            # executing SMOTE to generate some minority samples
            smote = SMOTE(proportion=prop,
                          n_neighbors=self.n_neighbors,
                          n_jobs=self.n_jobs,
                          random_state=self.random_state)
            X_samp, y_samp = smote.sample(X, y)
            X_samp = X_samp[len(X):]

            new_synth = []

            # computing POS membership values for the new samples
            pos_synth = pairwise_distances(X_min, X_samp, metric='l1')
            pos_synth = 1.0 - pos_synth
            pos_synth = pos_synth.clip(0, d)
            pos_synth = 1.0 - pos_synth

            # adding samples with POS membership smaller than gamma_S to the
            # minority set
            min_pos = np.min(pos_synth, axis=0)
            to_add = np.where(min_pos < gamma_S)[0]
            result_synth.extend(X_samp[to_add])
            new_synth.extend(X_samp[to_add])

            # checking the minimum POS values of the majority samples
            min_pos = np.min(pos_cache, axis=0)
            to_remove = np.where(min_pos < self.gamma_M)[0]

            # if the number of majority samples with POS membership smaller
            # than gamma_M is not extreme, then changing labels, otherwise
            # decreasing gamma_M
            if len(to_remove) > (len(X_maj) - len(X_min))/2:
                to_remove = np.array([])
                gamma_M = gamma_M*0.9
                _logger.info(self.__class__.__name__ + ": " +
                             "gamma_M decreased to %f" % gamma_M)
            else:
                result_maj.extend(X_maj[to_remove])
                X_maj = np.delete(X_maj, to_remove, axis=0)
                pos_cache = np.delete(pos_cache, to_remove, axis=1)

            # updating pos cache
            if len(new_synth) > 0:
                pos_cache_new = pairwise_distances(
                    np.vstack(new_synth), X_maj, metric='l1')
                pos_cache_new = 1.0 - pos_cache_new
                pos_cache_new = pos_cache_new.clip(0, d)
                pos_cache_new = 1.0 - pos_cache_new

                pos_cache = np.vstack([pos_cache, pos_cache_new])

            message = "minority added: %d, majority removed %d"
            message = message % (len(to_add), len(to_remove))
            _logger.info(self.__class__.__name__ + ":" + message)

            iteration = iteration + 1

        # packing the results
        X_res = np.vstack([X_maj, X_min])
        if len(result_synth) > 0:
            X_res = np.vstack([X_res, np.vstack(result_synth)])
        if len(result_maj) > 0:
            X_res = np.vstack([X_res, np.vstack(result_maj)])

        if len(X_maj) == 0:
            _logger.warning('All majority samples removed')
            return mmscaler.inverse_transform(X), y

        y_res_maj = np.repeat(self.maj_label, len(X_maj))
        n_y_res_min = len(X_min) + len(result_synth) + len(result_maj)
        y_res_min = np.repeat(self.min_label, n_y_res_min)
        y_res = np.hstack([y_res_maj, y_res_min])

        return mmscaler.inverse_transform(X_res), y_res

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'gamma_S': self.gamma_S,
                'gamma_M': self.gamma_M,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class AND_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{and_smote,
                             author = {Yun, Jaesub and Ha,
                                 Jihyun and Lee, Jong-Seok},
                             title = {Automatic Determination of Neighborhood
                                        Size in SMOTE},
                             booktitle = {Proceedings of the 10th International
                                            Conference on Ubiquitous
                                            Information Management and
                                            Communication},
                             series = {IMCOM '16},
                             year = {2016},
                             isbn = {978-1-4503-4142-4},
                             location = {Danang, Viet Nam},
                             pages = {100:1--100:8},
                             articleno = {100},
                             numpages = {8},
                             doi = {10.1145/2857546.2857648},
                             acmid = {2857648},
                             publisher = {ACM},
                             address = {New York, NY, USA},
                             keywords = {SMOTE, imbalanced learning, synthetic
                                            data generation},
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self, proportion=1.0, K=15, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            K (int): maximum number of nearest neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(K, "K", 2)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.K = K
        self.n_jobs = n_jobs
        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """

        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'K': [9, 15, 21]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        K = min([len(X_min), self.K])
        # find K nearest neighbors of all samples
        nn = NearestNeighbors(n_neighbors=K, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X)

        min_ind = np.where(y == self.min_label)[0]

        # Executing the algorithm
        kappa = []
        for i in range(len(min_ind)):
            regions_min = []
            regions_maj = []

            for j in range(1, K):
                # continueing if the label of the neighbors is minority
                if y[ind[min_ind[i]][j]] != self.min_label:
                    continue

                # region coordinates
                reg = np.hstack([min_ind[i], ind[min_ind[i]][j]])
                # compute corner points
                reg_min = np.min(X[reg])
                reg_max = np.max(X[reg])

                r_min = []
                r_maj = []
                # all the points in the region must be among the neighbors
                # what we do is counting how many of them are minority and
                # majority samples
                for k in ind[min_ind[i]][:(j+1)]:
                    if np.all(reg_min <= X[k]) and np.all(X[k] <= reg_max):
                        if y[k] == self.min_label:
                            r_min.append(k)
                        else:
                            r_maj.append(k)

                # appending the coordinates of points to the minority and
                # majority regions
                regions_min.append(r_min)
                regions_maj.append(r_maj)

            # taking the cumulative unions of minority and majority points
            for j in range(1, len(regions_min)):
                regions_min[j] = list(
                    set(regions_min[j]).union(set(regions_min[j-1])))
                regions_maj[j] = list(
                    set(regions_maj[j]).union(set(regions_maj[j-1])))

            # computing the lengths of the increasing minority and majority
            # sets
            regions_min = np.array([len(r) for r in regions_min])
            regions_maj = np.array([len(r) for r in regions_maj])

            # computing the precision of minority classification (all points
            # are supposed to be classified as minority)
            prec = regions_min/(regions_min + regions_maj)
            # taking the difference
            d = np.diff(prec, 1)
            # finding the biggest drop (+1 because diff reduces length, +1
            # because of indexing begins with 0)
            if len(d) == 0:
                k = 0
            else:
                k = np.argmin(d) + 2
            # appending the coordinate of the biggest drop as the ideal
            # neighborhood size note that k indices the minority neighbors
            kappa.append(k)

        # finding nearest minority neighbors of minority samples
        nn = NearestNeighbors(n_neighbors=max(kappa) + 1, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        if np.sum(kappa) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "No minority samples in nearest neighbors")
            return X.copy(), y.copy()

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # choose random point
            idx = self.random_state.randint(len(X_min))
            if kappa[idx] > 0:
                domain = ind[idx][1:(kappa[idx]+1)]
                X_b = X_min[self.random_state.choice(domain)]
                samples.append(self.sample_between_points(X_min[idx], X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'K': self.K,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class NRAS(OverSampling):
    """
    References:
        * BibTex::

            @article{nras,
                        title = "Noise Reduction A Priori Synthetic
                                    Over-Sampling for class imbalanced data
                                    sets",
                        journal = "Information Sciences",
                        volume = "408",
                        pages = "146 - 161",
                        year = "2017",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2017.04.046",
                        author = "William A. Rivera",
                        keywords = "NRAS, SMOTE, OUPS, Class imbalance,
                                        Classification"
                        }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 t=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors
            t (float): [0,1] fraction of n_neighbors as threshold
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(t, "t", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.t = t
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [5, 7, 9],
                                  't': [0.3, 0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardization is needed to make the range of the propensity scores
        # similar to that of the features
        mms = MinMaxScaler()
        X_trans = mms.fit_transform(X)

        # determining propensity scores using logistic regression
        lr = LogisticRegression(solver='lbfgs',
                                n_jobs=self.n_jobs,
                                random_state=self.random_state)
        lr.fit(X_trans, y)
        propensity = lr.predict_proba(X_trans)[:, np.where(
            lr.classes_ == self.min_label)[0][0]]

        X_min = X_trans[y == self.min_label]

        # adding propensity scores as a new feature
        X_new = np.column_stack([X_trans, propensity])
        X_min_new = X_new[y == self.min_label]

        # finding nearest neighbors of minority samples
        n_neighbors = min([len(X_new), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_new)
        dist, ind = nn.kneighbors(X_min_new)

        # do the sampling
        samples = []
        to_remove = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            # finding the number of minority neighbors
            t_hat = np.sum(y[ind[idx][1:]] == self.min_label)
            if t_hat < self.t*n_neighbors:
                # removing the minority point if the number of minority
                # neighbors is less then the threshold
                # to_remove indexes X_min
                if idx not in to_remove:
                    to_remove.append(idx)
                    # compensating the removal of the minority point
                    n_to_sample = n_to_sample + 1

                if len(to_remove) == len(X_min):
                    _logger.warning(self.__class__.__name__ + ": " +
                                    "all minority samples identified as noise")
                    return X.copy(), y.copy()
            else:
                # otherwise do the sampling
                X_b = X_trans[self.random_state.choice(ind[idx][1:])]
                samples.append(self.sample_between_points(X_min[idx], X_b))

        # remove noisy elements
        X_maj = X_trans[y == self.maj_label]
        X_min = np.delete(X_min, to_remove, axis=0)

        return (mms.inverse_transform(np.vstack([X_maj,
                                                 X_min,
                                                 np.vstack(samples)])),
                np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min)),
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                't': self.t,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class AMSCO(OverSampling):
    """
    References:
        * BibTex::

            @article{amsco,
                        title = "Adaptive multi-objective swarm fusion for
                                    imbalanced data classification",
                        journal = "Information Fusion",
                        volume = "39",
                        pages = "1 - 24",
                        year = "2018",
                        issn = "1566-2535",
                        doi = "https://doi.org/10.1016/j.inffus.2017.03.007",
                        author = "Jinyan Li and Simon Fong and Raymond K.
                                    Wong and Victor W. Chu",
                        keywords = "Swarm fusion, Swarm intelligence
                                    algorithm, Multi-objective, Crossover
                                    rebalancing, Imbalanced data
                                    classification"
                        }

    Notes:
        * It is not clear how the kappa threshold is used, I do use the RA
            score to drive all the evolution. Particularly:

            "In the last phase of each iteration, the average Kappa value
            in current non-inferior set is compare with the latest threshold
            value, the threshold is then increase further if the average value
            increases, and vice versa. By doing so, the non-inferior region
            will be progressively reduced as the Kappa threshold lifts up."

        I don't see why would the Kappa threshold lift up if the kappa
        thresholds are decreased if the average Kappa decreases ("vice versa").

        * Due to the interpretation of kappa threshold and the lack of detailed
            description of the SIS process, the implementation is not exactly
            what is described in the paper, but something very similar.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_memetic,
                  OverSampling.cat_uses_classifier]

    def __init__(self,
                 n_pop=5,
                 n_iter=15,
                 omega=0.1,
                 r1=0.1,
                 r2=0.1,
                 n_jobs=1,
                 classifier=DecisionTreeClassifier(random_state=2),
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_pop (int): size of populations
            n_iter (int): optimization steps
            omega (float): intertia of PSO
            r1 (float): force towards local optimum
            r2 (float): force towards global optimum
            n_jobs (int): number of parallel jobs
        """
        super().__init__()
        self.check_greater_or_equal(n_pop, "n_pop", 1)
        self.check_greater_or_equal(n_iter, "n_iter", 1)
        self.check_greater_or_equal(omega, "omega", 0)
        self.check_greater_or_equal(r1, "r1", 0)
        self.check_greater_or_equal(r2, "r2", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_pop = n_pop
        self.n_iter = n_iter
        self.omega = omega
        self.r1 = r1
        self.r2 = r2
        self.n_jobs = n_jobs
        self.classifier = classifier

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        # as the method is an overall optimization, 1 reasonable settings
        # should be enough

        classifiers = [DecisionTreeClassifier(random_state=2)]
        parameter_combinations = {'n_pop': [5],
                                  'n_iter': [15],
                                  'omega': [0.1],
                                  'r1': [0.1],
                                  'r2': [0.1],
                                  'classifier': classifiers}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_cross_val = min([4, len(X_min)])

        def fitness(X_min, X_maj):
            """
            Calculating fitness function

            Args:
                X_min (np.matrix): minority samples
                X_maj (np.matrix): majority samples

            Returns:
                float, float: kappa, accuracy
            """
            kfold = StratifiedKFold(n_cross_val)

            # prepare assembled dataset
            X_ass = np.vstack([X_min, X_maj])
            y_ass = np.hstack([np.repeat(self.min_label, len(X_min)),
                               np.repeat(self.maj_label, len(X_maj))])

            preds = []
            tests = []
            for train, test in kfold.split(X_ass, y_ass):
                self.classifier.fit(X_ass[train], y_ass[train])
                preds.append(self.classifier.predict(X))
                tests.append(y)
            preds = np.hstack(preds)
            tests = np.hstack(tests)

            # calculate kappa and accuracy scores
            tp = np.sum(np.logical_and(preds == tests,
                                       tests == self.min_label))
            fn = np.sum(np.logical_and(preds != tests,
                                       tests == self.min_label))
            tn = np.sum(np.logical_and(preds == tests,
                                       tests == self.maj_label))
            fp = np.sum(np.logical_and(preds != tests,
                                       tests == self.maj_label))

            p_o = (tp + tn)/(tp + fn + tn + fp)
            p_e = (tp + fn)*(tp + fp)/(tp + fn + tn + fp)**2 + \
                (fp + tn)*(fn + tn)/(tp + fn + tn + fp)**2

            kappa = (p_o - p_e)/(1.0 - p_e)
            accuracy = (tp + tn)/(tp + fn + tn + fp)

            return kappa, accuracy

        def OSMOTE(X_min, X_maj):
            """
            Executing OSMOTE phase

            Args:
                X_min (np.matrix): minority samples
                X_maj (np.matrix): majority samples

            Returns:
                np.matrix, np.matrix: new minority and majority datasets
            """

            # initialize particles, first coordinate represents proportion
            # parameter of SMOTE
            # the second coordinate represents the number of neighbors to
            # take into consideration
            def init_pop():
                proportion = self.random_state.random_sample()/2.0+0.5
                n_neighbors = self.random_state.randint(3, 10)
                return np.array([proportion, n_neighbors])
            particles = [init_pop() for _ in range(self.n_pop)]
            # velocities initialized
            velocities = [np.array([0.1, 1]) for _ in range(self.n_pop)]
            # setting the limits of the search space
            limits = [np.array([0.25, 3]), np.array([4.0, 10])]
            # local best results
            local_best = [particles[i].copy() for i in range(self.n_pop)]
            # local best scores
            local_score = [(0.0, 0.0)]*self.n_pop
            # global best result
            global_best = particles[0].copy()
            # global best score
            global_score = (0.0, 0.0)
            # best dataset
            best_dataset = None

            # running the optimization
            for _ in range(self.n_iter):
                # update velocities
                for i in range(len(velocities)):
                    diff1 = (local_best[i] - velocities[i])
                    diff2 = (global_best - velocities[i])
                    velocities[i] = (velocities[i]*self.omega +
                                     self.r1 * diff1 + self.r2*diff2)
                    # clipping velocities using the upper bounds of the
                    # particle search space
                    velocities[i][0] = np.clip(
                        velocities[i][0], -limits[1][0]/2, limits[1][0]/2)
                    velocities[i][1] = np.clip(
                        velocities[i][1], -limits[1][1]/2, limits[1][1]/2)

                # update particles
                for i in range(len(particles)):
                    particles[i] = particles[i] + velocities[i]
                    # clipping the particle positions using the lower and
                    # upper bounds
                    particles[i][0] = np.clip(
                        particles[i][0], limits[0][0], limits[1][0])
                    particles[i][1] = np.clip(
                        particles[i][1], limits[0][1], limits[1][1])

                # evaluate
                scores = []
                for i in range(len(particles)):
                    # apply SMOTE
                    smote = SMOTE(particles[i][0],
                                  int(np.rint(particles[i][1])),
                                  n_jobs=self.n_jobs,
                                  random_state=self.random_state)
                    X_to_sample = np.vstack([X_maj, X_min])
                    y_to_sample_maj = np.repeat(
                        self.maj_label, len(X_maj))
                    y_to_sample_min = np.repeat(
                        self.min_label, len(X_min))
                    y_to_sample = np.hstack([y_to_sample_maj, y_to_sample_min])
                    X_samp, y_samp = smote.sample(X_to_sample, y_to_sample)

                    # evaluate
                    scores.append(fitness(X_samp[len(X_maj):],
                                          X_samp[:len(X_maj)]))

                    # update scores according to the multiobjective setting
                    if (scores[i][0]*scores[i][1] >
                            local_score[i][0]*local_score[i][1]):
                        local_best[i] = particles[i].copy()
                        local_score[i] = scores[i]
                    if (scores[i][0]*scores[i][1] >
                            global_score[0]*global_score[1]):
                        global_best = particles[i].copy()
                        global_score = scores[i]
                        best_dataset = (X_samp[len(X_maj):],
                                        X_samp[:len(X_maj)])

            return best_dataset[0], best_dataset[1]

        def SIS(X_min, X_maj):
            """
            SIS procedure

            Args:
                X_min (np.matrix): minority dataset
                X_maj (np.matrix): majority dataset

            Returns:
                np.matrix, np.matrix: new minority and majority datasets
            """
            min_num = len(X_min)
            max_num = len(X_maj)
            if min_num >= max_num:
                return X_min, X_maj

            # initiate particles
            def init_particle():
                num = self.random_state.randint(min_num, max_num)
                maj = self.random_state.choice(np.arange(len(X_maj)), num)
                return maj

            particles = [init_particle() for _ in range(self.n_pop)]
            scores = [fitness(X_min, X_maj[particles[i]])
                      for i in range(self.n_pop)]
            best_score = (0.0, 0.0)
            best_dataset = None

            for _ in range(self.n_iter):
                # mutate and evaluate
                # the way mutation or applying PSO is not described in the
                # paper in details
                for i in range(self.n_pop):
                    # removing some random elements
                    domain = np.arange(len(particles[i]))
                    n_max = min([10, len(particles[i])])
                    n_to_choose = self.random_state.randint(0, n_max)
                    to_remove = self.random_state.choice(domain, n_to_choose)
                    mutant = np.delete(particles[i], to_remove)

                    # adding some random elements
                    maj_set = set(np.arange(len(X_maj)))
                    part_set = set(particles[i])
                    diff = list(maj_set.difference(part_set))
                    n_max = min([10, len(diff)])
                    n_to_choose = self.random_state.randint(0, n_max)
                    diff_elements = self.random_state.choice(diff, n_to_choose)
                    mutant = np.hstack([mutant, np.array(diff_elements)])
                    # evaluating the variant
                    score = fitness(X_min, X_maj[mutant])
                    if score[1] > scores[i][1]:
                        particles[i] = mutant.copy()
                        scores[i] = score
                    if score[1] > best_score[1]:
                        best_score = score
                        best_dataset = mutant.copy()

            return X_min, X_maj[best_dataset]

        # executing the main optimization procedure
        current_min = X_min
        current_maj = X_maj
        for it in range(self.n_iter):
            _logger.info(self.__class__.__name__ + ": " +
                         'staring iteration %d' % it)
            new_min, _ = OSMOTE(X_min, current_maj)
            _, new_maj = SIS(current_min, X_maj)

            # calculating fitness values of the four combinations
            fitness_0 = np.prod(fitness(new_min, current_maj))
            fitness_1 = np.prod(fitness(current_min, current_maj))
            fitness_2 = np.prod(fitness(new_min, new_maj))
            fitness_3 = np.prod(fitness(current_min, new_maj))

            # selecting the new current_maj and current_min datasets
            message = 'fitness scores: %f %f %f %f'
            message = message % (fitness_0, fitness_1, fitness_2, fitness_3)
            _logger.info(self.__class__.__name__ + ": " + message)
            max_fitness = np.max([fitness_0, fitness_1, fitness_2, fitness_3])
            if fitness_1 == max_fitness or fitness_3 == max_fitness:
                current_maj = new_maj
            if fitness_0 == max_fitness or fitness_2 == max_fitness:
                current_min = new_min

        return (np.vstack([current_maj, current_min]),
                np.hstack([np.repeat(self.maj_label, len(current_maj)),
                           np.repeat(self.min_label, len(current_min))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_pop': self.n_pop,
                'n_iter': self.n_iter,
                'omega': self.omega,
                'r1': self.r1,
                'r2': self.r2,
                'n_jobs': self.n_jobs,
                'classifier': self.classifier,
                'random_state': self._random_state_init}


class SSO(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{sso,
                            author="Rong, Tongwen
                            and Gong, Huachang
                            and Ng, Wing W. Y.",
                            editor="Wang, Xizhao
                            and Pedrycz, Witold
                            and Chan, Patrick
                            and He, Qiang",
                            title="Stochastic Sensitivity Oversampling
                                    Technique for Imbalanced Data",
                            booktitle="Machine Learning and Cybernetics",
                            year="2014",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="161--171",
                            isbn="978-3-662-45652-1"
                            }

    Notes:
        * In the algorithm step 2d adds a constant to a vector. I have
            changed it to a componentwise adjustment, and also used the
            normalized STSM as I don't see any reason why it would be
            some reasonable, bounded value.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 h=10,
                 n_iter=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            h (int): number of hidden units
            n_iter (int): optimization steps
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(h, "h", 1)
        self.check_greater_or_equal(n_iter, "n_iter", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.h = h
        self.n_iter = n_iter
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5],
                                  'h': [2, 5, 10, 20],
                                  'n_iter': [5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # number of samples to generate in each iteration
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        samp_per_iter = max([1, int(n_to_sample/self.n_iter)])

        # executing the algorithm
        for _ in range(self.n_iter):
            X_min = X[y == self.min_label]

            # applying kmeans clustering to find the hidden neurons
            h = min([self.h, len(X_min)])
            kmeans = KMeans(n_clusters=h,
                            random_state=self.random_state)
            kmeans.fit(X)

            # extracting the hidden center elements
            u = kmeans.cluster_centers_

            # extracting scale parameters as the distances of closest centers
            nn_cent = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
            nn_cent.fit(u)
            dist_cent, ind_cent = nn_cent.kneighbors(u)
            v = dist_cent[:, 1]

            # computing the response of the hidden units
            phi = pairwise_distances(X, u)
            phi = phi**2
            phi = np.exp(-phi/v**2)

            # applying linear regression to find the best weights
            lr = LinearRegression()
            lr.fit(phi, y)
            f = lr.predict(phi[np.where(y == self.min_label)[0]])
            w = lr.coef_

            def eq_6(Q, w, u, v, x):
                """
                Equation 6 in the paper
                """
                tmp_sum = np.zeros(h)
                for i in range(h):
                    a = (x - u[i] + Q)/np.sqrt(2*v[i])
                    b = (x - u[i] - Q)/np.sqrt(2*v[i])
                    tmp_prod = (sspecial.erf(a) - sspecial.erf(b))
                    tmp_sum[i] = np.sqrt(np.pi/2)*v[i]*np.prod(tmp_prod)
                return np.dot(tmp_sum, w)/(2*Q)**len(x)

            def eq_8(Q, w, u, v, x):
                """
                Equation 8 in the paper
                """
                res = 0.0
                for i in range(h):
                    vi2 = v[i]**2
                    for r in range(h):
                        vr2 = v[r]**2
                        a1 = (np.sqrt(2*vi2*vr2*(vi2 + vr2)))

                        a00_v = (vi2 + vr2)*(x + Q)
                        a01_v = vi2*u[r] + vr2*u[i]
                        a0_v = a00_v - a01_v
                        a_v = a0_v/a1

                        b_v = ((vi2 + vr2)*(x - Q) - (vi2*u[r] + vr2*u[i]))/a1
                        tmp_prod = sspecial.erf(a_v) - sspecial.erf(b_v)

                        tmp_a = (np.sqrt(2*vi2*vr2*(vi2 + vr2)) /
                                 (vi2 + vr2))**len(x)
                        norm = np.linalg.norm(u[r] - u[i])
                        tmp_b = np.exp(-0.5 * norm**2/(vi2 + vr2))
                        res = res + tmp_a*tmp_b*np.prod(tmp_prod)*w[i]*w[r]

                return (np.sqrt(np.pi)/(4*Q))**len(x)*res

            # applying nearest neighbors to extract Q values
            n_neighbors = min([self.n_neighbors + 1, len(X)])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            nn.fit(X)
            dist, ind = nn.kneighbors(X_min)

            Q = np.mean(dist[:, n_neighbors-1])/np.sqrt(len(X[0]))

            # calculating the sensitivity factors
            I_1 = np.array([eq_6(Q, w, u, v, x) for x in X_min])
            I_2 = np.array([eq_8(Q, w, u, v, x) for x in X_min])

            stsm = f**2 - 2*f*I_1 + I_2

            # calculating the sampling weights
            weights = np.abs(stsm)/np.sum(np.abs(stsm))

            n_neighbors = min([len(X_min), self.n_neighbors+1])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            nn.fit(X_min)
            dist, ind = nn.kneighbors(X_min)

            samples = []
            for _ in range(samp_per_iter):
                idx = self.random_state.choice(
                    np.arange(len(X_min)), p=weights)
                X_new = X_min[idx].copy()
                for s in range(len(X_new)):
                    lam = self.random_state.random_sample(
                    )*(2*(1 - weights[idx])) - (1 - weights[idx])
                    X_new[s] = X_new[s] + Q*lam
                samples.append(X_new)

            samples = np.vstack(samples)
            X = np.vstack([X, samples])
            y = np.hstack([y, np.repeat(self.min_label, len(samples))])

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'h': self.h,
                'n_iter': self.n_iter,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class NDO_sampling(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ndo_sampling,
                            author={Zhang, L. and Wang, W.},
                            booktitle={2011 International Conference of
                                        Information Technology, Computer
                                        Engineering and Management Sciences},
                            title={A Re-sampling Method for Class Imbalance
                                    Learning with Credit Data},
                            year={2011},
                            volume={1},
                            number={},
                            pages={393-397},
                            keywords={data handling;sampling methods;
                                        resampling method;class imbalance
                                        learning;credit rating;imbalance
                                        problem;synthetic minority
                                        over-sampling technique;sample
                                        distribution;synthetic samples;
                                        credit data set;Training;
                                        Measurement;Support vector machines;
                                        Logistics;Testing;Noise;Classification
                                        algorithms;class imbalance;credit
                                        rating;SMOTE;sample distribution},
                            doi={10.1109/ICM.2011.34},
                            ISSN={},
                            month={Sept}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_application]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 T=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            T (float): threshold parameter
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(T, "T", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.T = T
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'T': [0.5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model to find the neighbors of minority
        # samples among all elements
        n_neighbors = min([len(X), self.n_neighbors+1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        # calculating the distances between samples in the same and different
        # classes
        d_intra = []
        d_exter = []
        for i in range(len(X_min)):
            min_mask = np.where(y[ind[i][1:]] == self.min_label)[0]
            maj_mask = np.where(y[ind[i][1:]] == self.maj_label)[0]
            if len(min_mask) > 0:
                d_intra.append(np.mean(dist[i][1:][min_mask]))
            if len(maj_mask) > 0:
                d_exter.append(np.mean(dist[i][1:][maj_mask]))
        d_intra_mean = np.mean(np.array(d_intra))
        d_exter_mean = np.mean(np.array(d_exter))

        # calculating the alpha value
        alpha = d_intra_mean/d_exter_mean

        # deciding if SMOTE is enough
        if alpha < self.T:
            smote = SMOTE(self.proportion, random_state=self.random_state)
            return smote.sample(X, y)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            random_idx = self.random_state.choice(ind[idx][1:])
            # create sample close to the initial minority point
            samples.append(X_min[idx] + (X[random_idx] - X_min[idx])
                           * self.random_state.random_sample()/2.0)
            if y[random_idx] == self.min_label:
                # create another sample close to the neighboring minority point
                samples.append(X[random_idx] + (X_min[idx] - X[random_idx])
                               * self.random_state.random_sample()/2.0)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'T': self.T,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class RBFNeuron(RandomStateMixin):
    """
    This class abstracts a neuron of an RBF network
    """

    def __init__(self,
                 c,
                 Ib,
                 Ob,
                 ranges,
                 range_mins,
                 init_conn_mask,
                 init_conn_weights,
                 random_state=None):
        """
        Constructor of the neuron

        Args:
            c (np.array): center of the hidden unit
            Ib (float): upper bound on the absolute values of input weights
            Ob (float): upper bound on the absolute values of output weights
            ranges (np.array): ranges widths of parameters
            range_min (np.array): lower bounds of parameter ranges
            init_conn_mask (np.array): initial input connections
            init_conn_weights (np.array): initial weights of input connections
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        self.d = len(c)
        self.c = c
        self.Ib = Ib
        self.Ob = Ob
        self.init_conn_mask = init_conn_mask
        self.init_conn_weights = init_conn_weights
        self.ranges = ranges
        self.range_mins = range_mins

        self.set_random_state(random_state)

        self.beta = (self.random_state.random_sample()-0.5)*Ob
        self.mask = init_conn_mask
        self.input_weights = init_conn_weights
        self.r = self.random_state.random_sample()

    def clone(self):
        """
        Clones the neuron

        Returns:
            RBFNeuron: an identical neuron
        """
        r = RBFNeuron(self.c,
                      self.Ib,
                      self.Ob,
                      self.ranges,
                      self.range_mins,
                      self.init_conn_mask,
                      self.init_conn_weights,
                      random_state=self.random_state)
        r.beta = self.beta
        r.mask = self.mask.copy()
        r.input_weights = self.input_weights.copy()
        r.r = self.r

        return r

    def evaluate(self, X):
        """
        Evaluates the system on dataset X

        Args:
            X (np.matrix): dataset to evaluate on

        Returns:
            np.array: the output of the network
        """
        wX = X[:, self.mask]*self.input_weights
        term_exp = -np.linalg.norm(wX - self.c[self.mask], axis=1)**2/self.r**2
        return self.beta*np.exp(term_exp)

    def mutate(self):
        """
        Mutates the neuron
        """
        r = self.random_state.random_sample()
        if r < 0.2:
            # centre creep
            self.c = self.random_state.normal(self.c, self.r)
        elif r < 0.4:
            # radius creep
            tmp = self.random_state.normal(self.r, np.var(self.ranges))
            if tmp > 0:
                self.r = tmp
        elif r < 0.6:
            # randomize centers
            self.c = self.random_state.random_sample(
                size=len(self.c))*self.ranges + self.range_mins
        elif r < 0.8:
            # randomize radii
            self.r = self.random_state.random_sample()*np.mean(self.ranges)
        else:
            # randomize output weight
            self.beta = self.random_state.normal(self.beta, self.Ob)

    def add_connection(self):
        """
        Adds a random input connection to the neuron
        """
        if len(self.mask) < self.d:
            d_set = set(range(self.d))
            mask_set = set(self.mask.tolist())
            domain = list(d_set.difference(mask_set))
            additional_elements = np.array(self.random_state.choice(domain))
            self.mask = np.hstack([self.mask, additional_elements])
            random_weight = (self.random_state.random_sample()-0.5)*self.Ib
            self.input_weights = np.hstack([self.input_weights, random_weight])

    def delete_connection(self):
        """
        Deletes a random input connection
        """
        if len(self.mask) > 1:
            idx = self.random_state.randint(len(self.mask))
            self.mask = np.delete(self.mask, idx)
            self.input_weights = np.delete(self.input_weights, idx)


class RBF(RandomStateMixin):
    """
    RBF network abstraction
    """

    def __init__(self,
                 X,
                 m_min,
                 m_max,
                 Ib,
                 Ob,
                 init_conn_mask,
                 init_conn_weights,
                 random_state=None):
        """
        Initializes the RBF network

        Args:
            X (np.matrix): dataset to work with
            m_min (int): minimum number of hidden neurons
            m_max (int): maximum number of hidden neurons
            Ib (float): maximum absolute value of input weights
            Ob (float): maximum absolute value of output weights
            init_conn_mask (np.array): initial input connections
            init_conn_weights (np.array): initial input weights
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        self.X = X
        self.m_min = m_min
        self.m_max = m_max
        self.Ib = Ib
        self.Ob = Ob
        self.init_conn_mask = init_conn_mask
        self.init_conn_weights = init_conn_weights

        self.set_random_state(random_state)

        self.neurons = []
        self.range_mins = np.min(X, axis=0)
        self.ranges = np.max(X, axis=0) - self.range_mins

        # adding initial neurons
        num_neurons = self.random_state.randint(m_min, m_max)
        for _ in range(num_neurons):
            self.neurons.append(self.create_new_node())

        self.beta_0 = (self.random_state.random_sample()-0.5)*Ob

    def clone(self):
        """
        Clones the entire network

        Returns:
            RBF: the cloned network
        """
        r = RBF(self.X,
                self.m_min,
                self.m_max,
                self.Ib,
                self.Ob,
                self.init_conn_mask,
                self.init_conn_weights,
                random_state=self.random_state)
        r.neurons = [n.clone() for n in self.neurons]
        r.range_mins = self.range_mins.copy()
        r.ranges = self.ranges.copy()
        r.beta_0 = self.beta_0

        return r

    def create_new_node(self):
        """
        Creates a new node.

        Returns:
            RBFNeuron: a new hidden neuron
        """
        return RBFNeuron(self.X[self.random_state.randint(len(self.X))],
                         self.Ib,
                         self.Ob,
                         self.ranges,
                         self.range_mins,
                         self.init_conn_mask,
                         self.init_conn_weights,
                         random_state=self.random_state)

    def update_data(self, X):
        """
        Updates the data to work with
        """
        self.X = X
        for n in self.neurons:
            n.X = X

    def improve_centers(self):
        """
        Improves the center locations by kmeans clustering
        """
        if len(np.unique(self.X, axis=0)) > len(self.neurons):
            cluster_init = np.vstack([n.c for n in self.neurons])
            kmeans = KMeans(n_clusters=len(self.neurons),
                            init=cluster_init,
                            n_init=1,
                            max_iter=30,
                            random_state=self.random_state)
            kmeans.fit(self.X)
            for i in range(len(self.neurons)):
                self.neurons[i].c = kmeans.cluster_centers_[i]

    def evaluate(self, X, y):
        """
        Evaluates the target function

        Returns:
            float: the target function value
        """
        evaluation = np.column_stack([n.evaluate(X) for n in self.neurons])
        f = self.beta_0 + np.sum(evaluation, axis=1)
        L_star = np.mean(abs(y[y == 1] - f[y == 1]))
        L_star += np.mean(abs(y[y == 0] - f[y == 0]))
        return L_star

    def mutation(self):
        """
        Mutates the neurons

        Returns:
            RBF: a new, mutated RBF network
        """
        rbf = self.clone()
        for n in rbf.neurons:
            n.mutate()
        return rbf

    def structural_mutation(self):
        """
        Applies structural mutation

        Returns:
            RBF: a new, structurally mutated network
        """
        # in the binary case the removal of output connections is the same as
        # removing hidden nodes
        rbf = self.clone()
        r = self.random_state.random_sample()
        if r < 0.5:
            if len(rbf.neurons) < rbf.m_max:
                rbf.neurons.append(rbf.create_new_node())
            elif len(rbf.neurons) > rbf.m_min:
                del rbf.neurons[self.random_state.randint(len(rbf.neurons))]
        else:
            rbf.neurons[self.random_state.randint(
                len(rbf.neurons))].delete_connection()
            rbf.neurons[self.random_state.randint(
                len(rbf.neurons))].add_connection()

        return rbf

    def recombine(self, rbf):
        """
        Recombines two networks

        Args:
            rbf (RBF): another network

        Returns:
            RBF: the result of recombination
        """
        # the order of neurons doesn't matter, so the logic can be simplified
        new = self.clone()
        if self.random_state.random_sample() < 0.5:
            n_random = self.random_state.randint(1, len(new.neurons))
            new_neurons_0 = self.random_state.choice(new.neurons, n_random)
            n_random = self.random_state.randint(1, len(rbf.neurons))
            new_neurons_1 = self.random_state.choice(rbf.neurons, n_random)
            new.neurons = [n.clone() for n in new_neurons_0]
            new.neurons.extend([n.clone() for n in new_neurons_1])
            while len(new.neurons) > self.m_max:
                del new.neurons[self.random_state.randint(len(new.neurons))]
        else:
            for i in range(len(new.neurons)):
                if self.random_state.random_sample() < 0.2:
                    n_random = self.random_state.randint(len(rbf.neurons))
                    new.neurons[i] = rbf.neurons[n_random].clone()
        return new


class DSRBF(OverSampling):
    """
    References:
        * BibTex::

            @article{dsrbf,
                        title = "A dynamic over-sampling procedure based on
                                    sensitivity for multi-class problems",
                        journal = "Pattern Recognition",
                        volume = "44",
                        number = "8",
                        pages = "1821 - 1833",
                        year = "2011",
                        issn = "0031-3203",
                        doi = "https://doi.org/10.1016/j.patcog.2011.02.019",
                        author = "Francisco Fernández-Navarro and César
                                    Hervás-Martínez and Pedro Antonio
                                    Gutiérrez",
                        keywords = "Classification, Multi-class, Sensitivity,
                                    Accuracy, Memetic algorithm, Imbalanced
                                    datasets, Over-sampling method, SMOTE"
                        }

    Notes:
        * It is not entirely clear why J-1 output is supposed where J is the
            number of classes.
        * The fitness function is changed to a balanced mean loss, as I found
            that it just ignores classification on minority samples
            (class label +1) in the binary case.
        * The iRprop+ optimization is not implemented.
        * The original paper proposes using SMOTE incrementally. Instead of
            that, this implementation applies SMOTE to generate all samples
            needed in the sampling epochs and the evolution of RBF networks
            is used to select the sampling providing the best results.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_memetic]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 m_min=4,
                 m_max=10,
                 Ib=2,
                 Ob=2,
                 n_pop=500,
                 n_init_pop=5000,
                 n_iter=40,
                 n_sampling_epoch=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
            m_min (int): minimum number of hidden units
            m_max (int): maximum number of hidden units
            Ib (float): input weight range
            Ob (float): output weight range
            n_pop (int): size of population
            n_init_pop (int): size of initial population
            n_iter (int): number of iterations
            n_sampling_epoch (int): resampling after this many iterations
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(m_min, "m_min", 1)
        self.check_greater_or_equal(m_max, "m_max", 1)
        self.check_greater(Ib, "Ib", 0)
        self.check_greater(Ob, "Ob", 0)
        self.check_greater_or_equal(n_pop, "n_pop", 2)
        self.check_greater_or_equal(n_init_pop, "n_pop", 2)
        self.check_greater_or_equal(n_iter, "n_iter", 0)
        self.check_greater_or_equal(n_sampling_epoch, "n_sampling_epoch", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.m_min = m_min
        self.m_max = m_max
        self.Ib = Ib
        self.Ob = Ob
        self.n_pop = n_pop
        self.n_init_pop = n_init_pop
        self.n_iter = n_iter
        self.n_sampling_epoch = n_sampling_epoch
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        # as the technique optimizes, it is unnecessary to check various
        # combinations except one specifying a decent workspace with a large
        # number of iterations
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'm_min': [4],
                                  'm_max': [10],
                                  'Ib': [2.0],
                                  'Ob': [2.0],
                                  'n_pop': [100],
                                  'n_init_pop': [1000],
                                  'n_iter': [40],
                                  'n_sampling_epoch': [8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # Standardizing the data to let the network work with comparable
        # attributes
        ss = StandardScaler()
        X = ss.fit_transform(X)
        X_orig = X
        y_orig = y

        X, y = SMOTE(proportion=self.proportion,
                     n_neighbors=self.n_neighbors,
                     n_jobs=self.n_jobs,
                     random_state=self.random_state).sample(X, y)

        # generate initial connections and weights randomly
        domain = np.arange(len(X[0]))
        n_random = int(len(X[0])/2)
        init_conn_mask = self.random_state.choice(domain, n_random)
        init_conn_weights = self.random_state.random_sample(size=n_random)

        # setting epoch lengths
        epoch_len = int(self.n_iter/self.n_sampling_epoch)

        if len(X_orig) < self.m_min + 1:
            return X_orig.copy(), y_orig.copy()
        m_max = min(len(X_orig), self.m_max)

        # generating initial population
        def init_pop():
            return RBF(X,
                       self.m_min,
                       m_max,
                       self.Ib,
                       self.Ob,
                       init_conn_mask,
                       init_conn_weights,
                       random_state=self.random_state)

        population = [init_pop() for _ in range(self.n_init_pop)]
        population = [[p, X, y, np.inf] for p in population]
        population = sorted([[p[0], p[1], p[2], p[0].evaluate(p[1], p[2])]
                             for p in population], key=lambda x: x[3])
        population = population[:self.n_pop]

        # executing center improval in the hidden units
        for p in population:
            p[0].improve_centers()

        # executing the optimization process
        for iteration in range(self.n_iter):
            message = "Iteration %d/%d, loss: %f, data size %d"
            message = message % (iteration, self.n_iter, population[0][3],
                                 len(population[0][1]))
            _logger.info(self.__class__.__name__ + ": " + message)
            # evaluating non-evaluated elements
            for p in population:
                if p[3] == np.inf:
                    p[3] = p[0].evaluate(p[1], p[2])

            # sorting the population by the loss values
            population = sorted([p for p in population], key=lambda x: x[3])
            population = population[:self.n_pop]

            # determining the number of elements to be changed
            p_best = population[0]
            p_parametric_mut = population[:int(0.1*self.n_pop)]
            p_structural_mut = population[:int(0.9*self.n_pop-1)]
            p_recombination = population[:int(0.1*self.n_pop)]

            # executing mutation
            for p in p_parametric_mut:
                population.append([p[0].mutation(), p[1], p[2], np.inf])

            # executing structural mutation
            for p in p_structural_mut:
                population.append(
                    [p[0].structural_mutation(), p[1], p[2], np.inf])

            # executing recombination
            for p in p_recombination:
                domain = range(len(p_recombination))
                p_rec_idx = self.random_state.choice(domain)
                p_rec = p_recombination[p_rec_idx][0]
                population.append([p[0].recombine(p_rec), p[1], p[2], np.inf])

            # do the sampling
            if iteration % epoch_len == 0:
                smote = SMOTE(proportion=self.proportion,
                              n_neighbors=self.n_neighbors,
                              n_jobs=self.n_jobs,
                              random_state=self.random_state)
                X, y = smote.sample(X_orig, y_orig)
                for i in range(self.n_pop):
                    tmp = [population[i][0].clone(), X, y, np.inf]
                    tmp[0].update_data(X)
                    tmp[0].improve_centers()
                    population.append(tmp)

        # evaluate unevaluated elements of the population
        for p in population:
            if p[3] == np.inf:
                p[3] = p[0].evaluate(p[1], p[2])

        # sorting the population
        population = sorted([p for p in population],
                            key=lambda x: x[3])[:self.n_pop]

        return ss.inverse_transform(p_best[1]), p_best[2]

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'm_min': self.m_min,
                'm_max': self.m_max,
                'Ib': self.Ib,
                'Ob': self.Ob,
                'n_pop': self.n_pop,
                'n_init_pop': self.n_init_pop,
                'n_iter': self.n_iter,
                'n_sampling_epoch': self.n_sampling_epoch,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Gaussian_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{gaussian_smote,
                      title={Gaussian-Based SMOTE Algorithm for Solving Skewed
                                Class Distributions},
                      author={Hansoo Lee and Jonggeun Kim and Sungshin Kim},
                      journal={Int. J. Fuzzy Logic and Intelligent Systems},
                      year={2017},
                      volume={17},
                      pages={229-234}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 sigma=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            sigma (float): variance
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater(sigma, "sigma", 0.0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.sigma = sigma
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'sigma': [0.5, 1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # standardization applied to make sigma compatible with the data
        ss = StandardScaler()
        X_ss = ss.fit_transform(X)

        # fitting nearest neighbors model to find the minority neighbors of
        # minority samples
        X_min = X_ss[y == self.min_label]
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            random_neighbor = self.random_state.choice(ind[idx][1:])
            s0 = self.sample_between_points(X_min[idx], X_min[random_neighbor])
            samples.append(self.random_state.normal(s0, self.sigma))

        return (np.vstack([X, ss.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'sigma': self.sigma,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class kmeans_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{kmeans_smote,
                        title = "Improving imbalanced learning through a
                                    heuristic oversampling method based
                                    on k-means and SMOTE",
                        journal = "Information Sciences",
                        volume = "465",
                        pages = "1 - 20",
                        year = "2018",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2018.06.056",
                        author = "Georgios Douzas and Fernando Bacao and
                                    Felix Last",
                        keywords = "Class-imbalanced learning, Oversampling,
                                    Classification, Clustering, Supervised
                                    learning, Within-class imbalance"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_clusters=10,
                 irt=2.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors
            n_clusters (int): number of clusters
            irt (float): imbalanced ratio threshold
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_clusters", 1)
        self.check_greater_or_equal(irt, "irt", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.irt = irt
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_clusters': [2, 5, 10, 20, 50],
                                  'irt': [0.5, 0.8, 1.0, 1.5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # applying kmeans clustering to all data
        n_clusters = min([self.n_clusters, len(X)])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X)

        # extracting clusters
        labels = kmeans.labels_
        clusters = [np.where(labels == li)[0] for li in range(n_clusters)]

        # cluster filtering
        def cluster_filter(c):
            numerator = np.sum(y[c] == self.maj_label) + 1
            denominator = np.sum(y[c] == self.min_label) + 1
            n_minority = np.sum(y[c] == self.min_label)
            return numerator/denominator < self.irt and n_minority > 1

        filt_clusters = [c for c in clusters if cluster_filter(c)]

        if len(filt_clusters) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "number of clusters after filtering is 0")
            return X.copy(), y.copy()

        # Step 2 in the paper
        sparsity = []
        nearest_neighbors = []
        cluster_minority_ind = []
        for c in filt_clusters:
            # extract minority indices in the cluster
            minority_ind = c[y[c] == self.min_label]
            cluster_minority_ind.append(minority_ind)
            # compute distance matrix of minority samples in the cluster
            dm = pairwise_distances(X[minority_ind])
            min_count = len(minority_ind)
            # compute the average of distances
            avg_min_dist = (np.sum(dm) - dm.trace()) / \
                (len(minority_ind)**2 - len(minority_ind))
            # compute sparsity (Step 4)
            sparsity.append(avg_min_dist**len(X[0])/min_count)
            # extract the nearest neighbors graph
            n_neighbors = min([len(minority_ind), self.n_neighbors + 1])
            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
            nn.fit(X[minority_ind])
            nearest_neighbors.append(nn.kneighbors(X[minority_ind]))

        # Step 5 - compute density of sampling
        weights = sparsity/np.sum(sparsity)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # choose random cluster index and random minority element
            clust_ind = self.random_state.choice(
                np.arange(len(weights)), p=weights)
            idx = self.random_state.randint(
                len(cluster_minority_ind[clust_ind]))
            base_idx = cluster_minority_ind[clust_ind][idx]
            # choose random neighbor
            neighbor_cluster_indices = nearest_neighbors[clust_ind][1][idx][1:]
            domain = cluster_minority_ind[clust_ind][neighbor_cluster_indices]
            neighbor_idx = self.random_state.choice(domain)
            # sample
            X_a = X[base_idx]
            X_b = X[neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_clusters': self.n_clusters,
                'irt': self.irt,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class Supervised_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{supervised_smote,
                        author = {Hu, Jun AND He, Xue AND Yu, Dong-Jun AND
                                    Yang, Xi-Bei AND Yang, Jing-Yu AND Shen,
                                    Hong-Bin},
                        journal = {PLOS ONE},
                        publisher = {Public Library of Science},
                        title = {A New Supervised Over-Sampling Algorithm
                                    with Application to Protein-Nucleotide
                                    Binding Residue Prediction},
                        year = {2014},
                        month = {09},
                        volume = {9},
                        url = {https://doi.org/10.1371/journal.pone.0107676},
                        pages = {1-10},
                        number = {9},
                        doi = {10.1371/journal.pone.0107676}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_application]

    def __init__(self,
                 proportion=1.0,
                 th_lower=0.5,
                 th_upper=1.0,
                 classifier=RandomForestClassifier(n_estimators=50,
                                                   n_jobs=1,
                                                   random_state=5),
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            th_lower (float): lower bound of the confidence interval
            th_upper (float): upper bound of the confidence interval
            classifier (obj): classifier used to estimate class memberships
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_in_range(th_lower, "th_lower", [0, 1])
        self.check_in_range(th_upper, "th_upper", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.th_lower = th_lower
        self.th_upper = th_upper
        self.classifier = classifier
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        classifiers = [RandomForestClassifier(n_estimators=50,
                                              n_jobs=1,
                                              random_state=5)]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'th_lower': [0.3, 0.5, 0.8],
                                  'th_upper': [1.0],
                                  'classifier': classifiers}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # training the classifier
        self.classifier.fit(X, y)

        X_min = X[y == self.min_label]

        th_lower = self.th_lower

        # do the sampling
        samples = []
        n_trials = 1
        n_success = 1
        while len(samples) < n_to_sample:
            n_trials = n_trials + 1

            domain = range(len(X_min))
            x0, x1 = self.random_state.choice(domain, 2, replace=False)
            x0, x1 = X_min[x0], X_min[x1]
            sample = self.sample_between_points(x0, x1)
            probs = self.classifier.predict_proba(sample.reshape(1, -1))
            # extract probability
            class_column = np.where(self.classifier.classes_ == self.min_label)
            class_column = class_column[0][0]
            prob = probs[0][class_column]
            if prob >= th_lower and prob <= self.th_upper:
                samples.append(sample)
                n_success = n_success + 1

            # decreasing lower threshold if needed
            if n_success/n_trials < 0.02:
                th_lower = th_lower * 0.9
                n_success = 1
                n_trials = 1

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'th_lower': self.th_lower,
                'th_upper': self.th_upper,
                'classifier': self.classifier,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class SN_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{sn_smote,
                        author="Garc{\'i}a, V.
                        and S{\'a}nchez, J. S.
                        and Mart{\'i}n-F{\'e}lez, R.
                        and Mollineda, R. A.",
                        title="Surrounding neighborhood-based SMOTE for
                                learning from imbalanced data sets",
                        journal="Progress in Artificial Intelligence",
                        year="2012",
                        month="Dec",
                        day="01",
                        volume="1",
                        number="4",
                        pages="347--362",
                        issn="2192-6360",
                        doi="10.1007/s13748-012-0027-5",
                        url="https://doi.org/10.1007/s13748-012-0027-5"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (float): number of neighbors
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # the search for the k nearest centroid neighbors is limited for the
        # nearest 10*n_neighbors neighbors
        n_neighbors = min([self.n_neighbors*10, len(X_min)])
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # determining k nearest centroid neighbors
        ncn = np.zeros(shape=(len(X_min), self.n_neighbors)).astype(int)
        ncn_nums = np.zeros(len(X_min)).astype(int)

        # extracting nearest centroid neighbors
        for i in range(len(X_min)):
            # the first NCN neighbor is the first neighbor
            ncn[i, 0] = ind[i][1]

            # iterating through all neighbors and finding the one with smaller
            # centroid distance to X_min[i] than the previous set of neighbors
            n_cent = 1
            centroid = X_min[ncn[i, 0]]
            cent_dist = np.linalg.norm(centroid - X_min[i])
            j = 2
            while j < len(ind[i]) and n_cent < self.n_neighbors:
                new_cent_dist = np.linalg.norm(
                    (centroid + X_min[ind[i][j]])/(n_cent + 1) - X_min[i])

                # checking if new nearest centroid neighbor found
                if new_cent_dist < cent_dist:
                    centroid = centroid + X_min[ind[i][j]]
                    ncn[i, n_cent] = ind[i][j]
                    n_cent = n_cent + 1
                    cent_dist = new_cent_dist
                j = j + 1

            # registering the number of nearest centroid neighbors found
            ncn_nums[i] = n_cent

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            random_idx = self.random_state.randint(len(X_min))
            random_neighbor_idx = self.random_state.choice(
                ncn[random_idx][:ncn_nums[random_idx]])
            samples.append(self.sample_between_points(
                X_min[random_idx], X_min[random_neighbor_idx]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class CCR(OverSampling):
    """
    References:
        * BibTex::

            @article{ccr,
                    author = {Koziarski, Michał and Wozniak, Michal},
                    year = {2017},
                    month = {12},
                    pages = {727–736},
                    title = {CCR: A combined cleaning and resampling algorithm
                                for imbalanced data classification},
                    volume = {27},
                    journal = {International Journal of Applied Mathematics
                                and Computer Science}
                    }

    Notes:
        * Adapted from https://github.com/michalkoziarski/CCR
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 energy=1.0,
                 scaling=0.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            energy (float): energy parameter
            scaling (float): scaling factor
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(energy, "energy", 0)
        self.check_greater_or_equal(scaling, "scaling", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.energy = energy
        self.scaling = scaling
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'energy': [0.001, 0.0025, 0.005,
                                             0.01, 0.025, 0.05, 0.1,
                                             0.25, 0.5, 1.0, 2.5, 5.0,
                                             10.0, 25.0, 50.0, 100.0],
                                  'scaling': [0.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        def taxicab_sample(n, r):
            sample = []
            random_numbers = self.random_state.rand(n)

            for i in range(n):
                # spread = r - np.sum(np.abs(sample))
                spread = r
                if len(sample) > 0:
                    spread -= abs(sample[-1])
                sample.append(spread * (2 * random_numbers[i] - 1))

            return self.random_state.permutation(sample)

        minority = X[y == self.min_label]
        majority = X[y == self.maj_label]

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances = pairwise_distances(minority, majority, metric='l1')

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy
            r = 0.0
            sorted_distances = np.argsort(distances[i])
            current_majority = 0

            while True:
                if current_majority > len(majority):
                    break

                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change = remaining_energy / \
                            (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change
                    break

                radius_change = remaining_energy / (current_majority + 1.0)

                dist = distances[i, sorted_distances[current_majority]]
                if dist >= r + radius_change:
                    r += radius_change
                    break
                else:
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        cm1 = current_majority - 1
                        last_distance = distances[i, sorted_distances[cm1]]

                    curr_maj_idx = sorted_distances[current_majority]
                    radius_change = distances[i, curr_maj_idx] - last_distance
                    r += radius_change
                    decrease = radius_change * (current_majority + 1.0)
                    remaining_energy -= decrease
                    current_majority += 1

            radii[i] = r

            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]].astype(float)
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    n_maj_point = len(majority_point)
                    r_num = self.random_state.rand(n_maj_point)
                    r_num = 1e-6 * r_num + 1e-6
                    r_sign = self.random_state.choice([-1.0, 1.0], n_maj_point)
                    majority_point += r_num * r_sign
                    d = np.sum(np.abs(minority_point - majority_point))

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority = majority.astype(float)
        majority += translations

        appended = []
        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = n_to_sample / (radii[i] * np.sum(1.0 / radii))
            synthetic_samples = int(np.round(synthetic_samples))
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point +
                                taxicab_sample(len(minority_point), r))

        if len(appended) == 0:
            _logger.info("No samples were added")
            return X.copy(), y.copy()

        return (np.vstack([X, np.vstack(appended)]),
                np.hstack([y, np.repeat(self.min_label, len(appended))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'energy': self.energy,
                'scaling': self.scaling,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class ANS(OverSampling):
    """
    References:
        * BibTex::

            @article{ans,
                     author = {Siriseriwan, W and Sinapiromsaran, Krung},
                     year = {2017},
                     month = {09},
                     pages = {565-576},
                     title = {Adaptive neighbor synthetic minority oversampling
                                technique under 1NN outcast handling},
                     volume = {39},
                     booktitle = {Songklanakarin Journal of Science and
                                    Technology}
                     }

    Notes:
        * The method is not prepared for the case when there is no c satisfying
            the condition in line 25 of the algorithm, fixed.
        * The method is not prepared for empty Pused sets, fixed.
    """
    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_density_based]

    def __init__(self, proportion=1.0, n_jobs=1, random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # outcast extraction algorithm

        # maximum C value
        C_max = int(0.25*len(X))

        # finding the first minority neighbor of minority samples
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # extracting the distances of first minority neighbors from minority
        # samples
        first_pos_neighbor_distances = dist[:, 1]

        # fitting another nearest neighbors model to extract majority
        # samples in the neighborhoods of minority samples
        nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
        nn.fit(X)

        # extracting the number of majority samples in the neighborhood of
        # minority samples
        out_border = []
        for i in range(len(X_min)):
            x = X_min[i].reshape(1, -1)
            ind = nn.radius_neighbors(x,
                                      first_pos_neighbor_distances[i],
                                      return_distance=False)
            out_border.append(np.sum(y[ind[0]] == self.maj_label))

        out_border = np.array(out_border)

        # finding the optimal C value by comparing the number of outcast
        # minority samples when traversing the range [1, C_max]
        n_oc_m1 = -1
        C = 0
        best_diff = np.inf
        for c in range(1, C_max):
            n_oc = np.sum(out_border >= c)
            if abs(n_oc - n_oc_m1) < best_diff:
                best_diff = abs(n_oc - n_oc_m1)
                C = n_oc
            n_oc_m1 = n_oc

        # determining the set of minority samples Pused
        Pused = np.where(out_border < C)[0]

        # Adaptive neighbor SMOTE algorithm

        # checking if there are minority samples left
        if len(Pused) == 0:
            _logger.info(self.__class__.__name__ + ": " + "Pused is empty")
            return X.copy(), y.copy()

        # finding the maximum distances of first positive neighbors
        eps = np.max(first_pos_neighbor_distances[Pused])

        # fitting nearest neighbors model to find nearest minority samples in
        # the neighborhoods of minority samples
        nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs)
        nn.fit(X_min[Pused])
        ind = nn.radius_neighbors(X_min[Pused], eps, return_distance=False)

        # extracting the number of positive samples in the neighborhoods
        Np = np.array([len(i) for i in ind])

        if np.all(Np == 1):
            message = "all samples have only 1 neighbor in the given radius"
            _logger.warning(self.__class__.__name__ + ": " + message)
            return X.copy(), y.copy()

        # determining the distribution used to generate samples
        distribution = Np/np.sum(Np)

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            random_idx = self.random_state.choice(
                np.arange(len(Pused)), p=distribution)
            if len(ind[random_idx]) > 1:
                random_neig_idx = self.random_state.choice(ind[random_idx])
                while random_neig_idx == random_idx:
                    random_neig_idx = self.random_state.choice(ind[random_idx])
                X_a = X_min[Pused[random_idx]]
                X_b = X_min[Pused[random_neig_idx]]
                samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


class cluster_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{cluster_SMOTE,
                            author={Cieslak, D. A. and Chawla, N. V. and
                                        Striegel, A.},
                            booktitle={2006 IEEE International Conference
                                        on Granular Computing},
                            title={Combating imbalance in network
                                        intrusion datasets},
                            year={2006},
                            volume={},
                            number={},
                            pages={732-737},
                            keywords={Intelligent networks;Intrusion detection;
                                        Telecommunication traffic;Data mining;
                                        Computer networks;Data security;
                                        Machine learning;Counting circuits;
                                        Computer security;Humans},
                            doi={10.1109/GRC.2006.1635905},
                            ISSN={},
                            month={May}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=3,
                 n_clusters=3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE
            n_clusters (int): number of clusters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_components", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.

        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_clusters': [3, 5, 7, 9]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class paramters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        X_min = X[y == self.min_label]

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        n_clusters = min([len(X_min), self.n_clusters])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state)
        kmeans.fit(X_min)
        cluster_labels = kmeans.labels_
        unique_labels = np.unique(cluster_labels)

        # creating nearest neighbors objects for each cluster
        cluster_indices = [np.where(cluster_labels == c)[0]
                           for c in unique_labels]

        def nneighbors(idx):
            n_neighbors = min([self.n_neighbors, len(cluster_indices[idx])])
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            return nn.fit(X_min[cluster_indices[idx]])

        cluster_nns = [nneighbors(idx) for idx in range(len(cluster_indices))]

        if max([len(c) for c in cluster_indices]) <= 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "All clusters contain 1 element")
            return X.copy(), y.copy()

        # generating the samples
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.randint(len(cluster_indices))
            if len(cluster_indices[cluster_idx]) <= 1:
                continue
            random_idx = self.random_state.randint(
                len(cluster_indices[cluster_idx]))
            sample_a = X_min[cluster_indices[cluster_idx]][random_idx]
            dist, indices = cluster_nns[cluster_idx].kneighbors(
                sample_a.reshape(1, -1))
            sample_b_idx = self.random_state.choice(
                cluster_indices[cluster_idx][indices[0][1:]])
            sample_b = X_min[sample_b_idx]
            samples.append(self.sample_between_points(sample_a, sample_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_clusters': self.n_clusters,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}


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


class OversamplingClassifier(BaseEstimator, ClassifierMixin):
    """
    This class wraps an oversampler and a classifier, making it compatible
    with sklearn based pipelines.
    """

    def __init__(self, oversampler, classifier):
        """
        Constructor of the wrapper.

        Args:
            oversampler (obj): an oversampler object
            classifier (obj): an sklearn-compatible classifier
        """

        self.oversampler = oversampler
        self.classifier = classifier

    def fit(self, X, y=None):
        """
        Carries out oversampling and fits the classifier.

        Args:
            X (np.ndarray): feature vectors
            y (np.array): target values

        Returns:
            obj: the object itself
        """

        X_samp, y_samp = self.oversampler.sample(X, y)
        self.classifier.fit(X_samp, y_samp)

        return self

    def predict(self, X):
        """
        Carries out the predictions.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Carries out the predictions with probability estimations.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict_proba(X)

    def get_params(self, deep=True):
        """
        Returns the dictionary of parameters.

        Args:
            deep (bool): wether to return parameters with deep discovery

        Returns:
            dict: the dictionary of parameters
        """

        return {'oversampler': self.oversampler, 'classifier': self.classifier}

    def set_params(self, **parameters):
        """
        Sets the parameters.

        Args:
            parameters (dict): the parameters to set.

        Returns:
            obj: the object itself
        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self


class MLPClassifierWrapper:
    """
    Wrapper over MLPClassifier of sklearn to provide easier parameterization
    """

    def __init__(self,
                 activation='relu',
                 hidden_layer_fraction=0.1,
                 alpha=0.0001,
                 random_state=None):
        """
        Constructor of the MLPClassifier

        Args:
            activation (str): name of the activation function
            hidden_layer_fraction (float): fraction of the hidden neurons of
                                            the number of input dimensions
            alpha (float): alpha parameter of the MLP classifier
            random_state (int/np.random.RandomState/None): initializer of the
                                                            random state
        """
        self.activation = activation
        self.hidden_layer_fraction = hidden_layer_fraction
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the model to the data

        Args:
            X (np.ndarray): features
            y (np.array): target labels

        Returns:
            obj: the MLPClassifierWrapper object
        """
        hidden_layer_size = max([1, int(len(X[0])*self.hidden_layer_fraction)])
        self.model = MLPClassifier(activation=self.activation,
                                   hidden_layer_sizes=(hidden_layer_size,),
                                   alpha=self.alpha,
                                   random_state=self.random_state).fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the labels of the unseen data

        Args:
            X (np.ndarray): unseen features

        Returns:
            np.array: predicted labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts the class probabilities of the unseen data

        Args:
            X (np.ndarray): unseen features

        Returns:
            np.matrix: predicted class probabilities
        """
        return self.model.predict_proba(X)

    def get_params(self, deep=False):
        """
        Returns the parameters of the classifier.

        Returns:
            dict: the parameters of the object
        """
        return {'activation': self.activation,
                'hidden_layer_fraction': self.hidden_layer_fraction,
                'alpha': self.alpha,
                'random_state': self.random_state}

    def copy(self):
        """
        Creates a copy of the classifier.

        Returns:
            obj: a copy of the classifier
        """
        return MLPClassifierWrapper(**self.get_params())


class Folding():
    """
    Cache-able folding of dataset for cross-validation
    """

    def __init__(self, dataset, validator, cache_path=None, random_state=None):
        """
        Constructor of Folding object

        Args:
            dataset (dict): dataset dictionary with keys 'data', 'target'
                            and 'DESCR'
            validator (obj): cross-validator object
            cache_path (str): path to cache directory
            random_state (int/np.random.RandomState/None): initializer of
                                                            the random state
        """
        self.dataset = dataset
        self.db_name = self.dataset['name']
        self.validator = validator
        self.cache_path = cache_path
        self.filename = 'folding_' + self.db_name + '.pickle'
        self.db_size = len(dataset['data'])
        self.db_n_attr = len(dataset['data'][0])
        self.imbalanced_ratio = np.sum(
            self.dataset['target'] == 0)/np.sum(self.dataset['target'] == 1)
        self.random_state = random_state

    def do_folding(self):
        """
        Does the folding or reads it from file if already available

        Returns:
            list(tuple): list of tuples of X_train, y_train, X_test, y_test
                            objects
        """

        self.validator.random_state = self.random_state

        if not hasattr(self, 'folding'):
            cond_cache_none = self.cache_path is None
            if not cond_cache_none:
                filename = os.path.join(self.cache_path, self.filename)
                cond_file_not_exists = not os.path.isfile(filename)
            else:
                cond_file_not_exists = False

            if cond_cache_none or cond_file_not_exists:
                _logger.info(self.__class__.__name__ +
                             (" doing folding %s" % self.filename))

                self.folding = {}
                self.folding['folding'] = []
                self.folding['db_size'] = len(self.dataset['data'])
                self.folding['db_n_attr'] = len(self.dataset['data'][0])
                n_maj = np.sum(self.dataset['target'] == 0)
                n_min = np.sum(self.dataset['target'] == 1)
                self.folding['imbalanced_ratio'] = n_maj / n_min

                X = self.dataset['data']
                y = self.dataset['target']

                data = self.dataset['data']
                target = self.dataset['target']

                for train, test in self.validator.split(data, target, target):
                    folding = (X[train], y[train], X[test], y[test])
                    self.folding['folding'].append(folding)
                if self.cache_path is not None:
                    _logger.info(self.__class__.__name__ +
                                 (" dumping to file %s" % self.filename))
                    random_filename = np.random.randint(1000000)
                    random_filename = str(random_filename) + '.pickle'
                    random_filename = os.path.join(self.cache_path,
                                                   random_filename)
                    pickle.dump(self.folding, open(random_filename, "wb"))
                    os.rename(random_filename, os.path.join(
                        self.cache_path, self.filename))
            else:
                _logger.info(self.__class__.__name__ +
                             (" reading from file %s" % self.filename))
                self.folding = pickle.load(
                    open(os.path.join(self.cache_path, self.filename), "rb"))
        return self.folding

    def get_params(self, deep=False):
        return {'db_name': self.db_name}

    def descriptor(self):
        return str(self.get_params())


class Sampling():
    """
    Cache-able sampling of dataset folds
    """

    def __init__(self,
                 folding,
                 sampler,
                 sampler_parameters,
                 scaler,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            folding (obj): Folding object
            sampler (class): class of a sampler object
            sampler_parameters (dict): a parameter combination for the sampler
                                        object
            scaler (obj): scaler object
            random_state (int/np.random.RandomState/None): initializer of the
                                                            random state
        """
        self.folding = folding
        self.db_name = folding.db_name
        self.sampler = sampler
        self.sampler_parameters = sampler_parameters
        self.sampler_parameters['random_state'] = random_state
        self.scaler = scaler
        self.cache_path = folding.cache_path
        self.filename = self.standardized_filename('sampling')
        self.random_state = random_state

    def standardized_filename(self,
                              prefix,
                              db_name=None,
                              sampler=None,
                              sampler_parameters=None):
        """
        standardizes the filename

        Args:
            filename (str): filename

        Returns:
            str: standardized name
        """
        import hashlib

        db_name = (db_name or self.db_name)

        sampler = (sampler or self.sampler)
        sampler = sampler.__name__
        sampler_parameters = sampler_parameters or self.sampler_parameters
        _logger.info(str(sampler_parameters))
        from collections import OrderedDict
        sampler_parameters_ordered = OrderedDict()
        for k in sorted(list(sampler_parameters.keys())):
            sampler_parameters_ordered[k] = sampler_parameters[k]

        message = " sampler parameter string "
        message = message + str(sampler_parameters_ordered)
        _logger.info(self.__class__.__name__ + message)
        sampler_parameter_str = hashlib.md5(
            str(sampler_parameters_ordered).encode('utf-8')).hexdigest()

        filename = '_'.join(
            [prefix, db_name, sampler, sampler_parameter_str]) + '.pickle'
        filename = re.sub('["\\,:(){}]', '', filename)
        filename = filename.replace("'", '')
        filename = filename.replace(": ", "_")
        filename = filename.replace(" ", "_")
        filename = filename.replace("\n", "_")

        return filename

    def cache_sampling(self):
        try:
            import mkl
            mkl.set_num_threads(1)
            _logger.info(self.__class__.__name__ +
                         (" mkl thread number set to 1 successfully"))
        except Exception as e:
            _logger.info(self.__class__.__name__ +
                         (" setting mkl thread number didn't succeed"))
            _logger.info(str(e))

        if not os.path.isfile(os.path.join(self.cache_path, self.filename)):
            # if the sampled dataset does not exist
            sampler_categories = self.sampler.categories
            is_extensive = OverSampling.cat_extensive in sampler_categories
            has_proportion = 'proportion' in self.sampler_parameters
            higher_prop_sampling_avail = None

            if is_extensive and has_proportion:
                proportion = self.sampler_parameters['proportion']
                all_pc = self.sampler.parameter_combinations()
                all_proportions = np.unique([p['proportion'] for p in all_pc])
                all_proportions = all_proportions[all_proportions > proportion]

                for p in all_proportions:
                    tmp_par = self.sampler_parameters.copy()
                    tmp_par['proportion'] = p
                    tmp_filename = self.standardized_filename(
                        'sampling', self.db_name, self.sampler, tmp_par)

                    filename = os.path.join(self.cache_path, tmp_filename)
                    if os.path.isfile(filename):
                        higher_prop_sampling_avail = (p, tmp_filename)
                        break

            if (not is_extensive or not has_proportion or
                    (is_extensive and has_proportion and
                        higher_prop_sampling_avail is None)):
                _logger.info(self.__class__.__name__ + " doing sampling")
                begin = time.time()
                sampling = []
                folds = self.folding.do_folding()
                for X_train, y_train, X_test, y_test in folds['folding']:
                    s = self.sampler(**self.sampler_parameters)

                    if self.scaler is not None:
                        print(self.scaler.__class__.__name__)
                        X_train = self.scaler.fit_transform(X_train, y_train)
                    X_samp, y_samp = s.sample_with_timing(X_train, y_train)

                    if hasattr(s, 'transform'):
                        X_test_trans = s.preprocessing_transform(X_test)
                    else:
                        X_test_trans = X_test.copy()

                    if self.scaler is not None:
                        X_samp = self.scaler.inverse_transform(X_samp)

                    sampling.append((X_samp, y_samp, X_test_trans, y_test))
                runtime = time.time() - begin
            else:
                higher_prop, higher_prop_filename = higher_prop_sampling_avail
                message = " reading and resampling from file %s to %s"
                message = message % (higher_prop_filename, self.filename)
                _logger.info(self.__class__.__name__ + message)
                filename = os.path.join(self.cache_path, higher_prop_filename)
                tmp_results = pickle.load(open(filename, 'rb'))
                tmp_sampling = tmp_results['sampling']
                tmp_runtime = tmp_results['runtime']

                sampling = []
                folds = self.folding.do_folding()
                nums = [len(X_train) for X_train, _, _, _ in folds['folding']]
                i = 0
                for X_train, y_train, X_test, y_test in tmp_sampling:
                    new_num = (len(X_train) - nums[i])/higher_prop*proportion
                    new_num = int(new_num)
                    offset = nums[i] + new_num
                    X_offset = X_train[:offset]
                    y_offset = y_train[:offset]
                    sampling.append((X_offset, y_offset, X_test, y_test))
                    i = i + 1
                runtime = tmp_runtime/p*proportion

            results = {}
            results['sampling'] = sampling
            results['runtime'] = runtime
            results['db_size'] = folds['db_size']
            results['db_n_attr'] = folds['db_n_attr']
            results['imbalanced_ratio'] = folds['imbalanced_ratio']

            _logger.info(self.__class__.__name__ +
                         (" dumping to file %s" % self.filename))

            random_filename = np.random.randint(1000000)
            random_filename = str(random_filename) + '.pickle'
            random_filename = os.path.join(self.cache_path, random_filename)
            pickle.dump(results, open(random_filename, "wb"))
            os.rename(random_filename, os.path.join(
                self.cache_path, self.filename))

    def do_sampling(self):
        self.cache_sampling()
        results = pickle.load(
            open(os.path.join(self.cache_path, self.filename), 'rb'))
        return results

    def get_params(self, deep=False):
        return {'folding': self.folding.get_params(),
                'sampler_name': self.sampler.__name__,
                'sampler_parameters': self.sampler_parameters}

    def descriptor(self):
        return str(self.get_params())


class Evaluation():
    """
    Cache-able evaluation of classifier on sampling
    """

    def __init__(self,
                 sampling,
                 classifiers,
                 n_threads=None,
                 random_state=None):
        """
        Constructor of an Evaluation object

        Args:
            sampling (obj): Sampling object
            classifiers (list(obj)): classifier objects
            n_threads (int/None): number of threads
            random_state (int/np.random.RandomState/None): random state
                                                            initializer
        """
        self.sampling = sampling
        self.classifiers = classifiers
        self.n_threads = n_threads
        self.cache_path = sampling.cache_path
        self.filename = self.sampling.standardized_filename('eval')
        self.random_state = random_state

        self.labels = []
        for i in range(len(classifiers)):
            from collections import OrderedDict
            sampling_parameters = OrderedDict()
            sp = self.sampling.sampler_parameters
            for k in sorted(list(sp.keys())):
                sampling_parameters[k] = sp[k]
            cp = classifiers[i].get_params()
            classifier_parameters = OrderedDict()
            for k in sorted(list(cp.keys())):
                classifier_parameters[k] = cp[k]

            label = str((self.sampling.db_name, sampling_parameters,
                         classifiers[i].__class__.__name__,
                         classifier_parameters))
            self.labels.append(label)

        print(self.labels)

    def calculate_metrics(self, all_pred, all_test, all_folds):
        """
        Calculates metrics of binary classifiction

        Args:
            all_pred (np.matrix): predicted probabilities
            all_test (np.matrix): true labels

        Returns:
            dict: all metrics of binary classification
        """

        results = {}
        if all_pred is not None:
            all_pred_labels = np.apply_along_axis(
                lambda x: np.argmax(x), 1, all_pred)

            results['tp'] = np.sum(np.logical_and(
                np.equal(all_test, all_pred_labels), (all_test == 1)))
            results['tn'] = np.sum(np.logical_and(
                np.equal(all_test, all_pred_labels), (all_test == 0)))
            results['fp'] = np.sum(np.logical_and(np.logical_not(
                np.equal(all_test, all_pred_labels)), (all_test == 0)))
            results['fn'] = np.sum(np.logical_and(np.logical_not(
                np.equal(all_test, all_pred_labels)), (all_test == 1)))
            results['p'] = results['tp'] + results['fn']
            results['n'] = results['fp'] + results['tn']
            results['acc'] = (results['tp'] + results['tn']) / \
                (results['p'] + results['n'])
            results['sens'] = results['tp']/results['p']
            results['spec'] = results['tn']/results['n']
            results['ppv'] = results['tp']/(results['tp'] + results['fp'])
            results['npv'] = results['tn']/(results['tn'] + results['fn'])
            results['fpr'] = 1.0 - results['spec']
            results['fdr'] = 1.0 - results['ppv']
            results['fnr'] = 1.0 - results['sens']
            results['bacc'] = (results['tp']/results['p'] +
                               results['tn']/results['n'])/2.0
            results['gacc'] = np.sqrt(
                results['tp']/results['p']*results['tn']/results['n'])
            results['f1'] = 2*results['tp'] / \
                (2*results['tp'] + results['fp'] + results['fn'])
            mcc_num = results['tp']*results['tn'] - results['fp']*results['fn']
            mcc_denom_0 = (results['tp'] + results['fp'])
            mcc_denom_1 = (results['tp'] + results['fn'])
            mcc_denom_2 = (results['tn'] + results['fp'])
            mcc_denom_3 = (results['tn'] + results['fn'])
            mcc_denom = mcc_denom_0 * mcc_denom_1 * mcc_denom_2*mcc_denom_3
            results['mcc'] = mcc_num/np.sqrt(mcc_denom)
            results['l'] = (results['p'] + results['n']) * \
                np.log(results['p'] + results['n'])
            tp_fp = (results['tp'] + results['fp'])
            tp_fn = (results['tp'] + results['fn'])
            tn_fp = (results['fp'] + results['tn'])
            tn_fn = (results['fn'] + results['tn'])
            results['ltp'] = results['tp']*np.log(results['tp']/(tp_fp*tp_fn))
            results['lfp'] = results['fp']*np.log(results['fp']/(tp_fp*tn_fp))
            results['lfn'] = results['fn']*np.log(results['fn']/(tp_fn*tn_fn))
            results['ltn'] = results['tn']*np.log(results['tn']/(tn_fp*tn_fn))
            results['lp'] = results['p'] * \
                np.log(results['p']/(results['p'] + results['n']))
            results['ln'] = results['n'] * \
                np.log(results['n']/(results['p'] + results['n']))
            uc_num = (results['l'] + results['ltp'] + results['lfp'] +
                      results['lfn'] + results['ltn'])
            uc_denom = (results['l'] + results['lp'] + results['ln'])
            results['uc'] = uc_num/uc_denom
            results['informedness'] = results['sens'] + results['spec'] - 1.0
            results['markedness'] = results['ppv'] + results['npv'] - 1.0
            results['log_loss'] = log_loss(all_test, all_pred)
            results['auc'] = roc_auc_score(all_test, all_pred[:, 1])
            aucs = [roc_auc_score(all_test[all_folds == i],
                                  all_pred[all_folds == i, 1])
                    for i in range(np.max(all_folds)+1)]
            results['auc_mean'] = np.mean(aucs)
            results['auc_std'] = np.std(aucs)
            test_labels, preds = zip(
                *sorted(zip(all_test, all_pred[:, 1]), key=lambda x: -x[1]))
            test_labels = np.array(test_labels)
            th = int(0.2*len(test_labels))
            results['p_top20'] = np.sum(test_labels[:th] == 1)/th
            results['brier'] = np.mean((all_pred[:, 1] - all_test)**2)
        else:
            results['tp'] = 0
            results['tn'] = 0
            results['fp'] = 0
            results['fn'] = 0
            results['p'] = 0
            results['n'] = 0
            results['acc'] = 0
            results['sens'] = 0
            results['spec'] = 0
            results['ppv'] = 0
            results['npv'] = 0
            results['fpr'] = 1
            results['fdr'] = 1
            results['fnr'] = 1
            results['bacc'] = 0
            results['gacc'] = 0
            results['f1'] = 0
            results['mcc'] = np.nan
            results['l'] = np.nan
            results['ltp'] = np.nan
            results['lfp'] = np.nan
            results['lfn'] = np.nan
            results['ltn'] = np.nan
            results['lp'] = np.nan
            results['ln'] = np.nan
            results['uc'] = np.nan
            results['informedness'] = 0
            results['markedness'] = 0
            results['log_loss'] = np.nan
            results['auc'] = 0
            results['auc_mean'] = 0
            results['auc_std'] = 0
            results['p_top20'] = 0
            results['brier'] = 1

        return results

    def do_evaluation(self):
        """
        Does the evaluation or reads it from file

        Returns:
            dict: all metrics
        """

        if self.n_threads is not None:
            try:
                import mkl
                mkl.set_num_threads(self.n_threads)
                message = " mkl thread number set to %d successfully"
                message = message % self.n_threads
                _logger.info(self.__class__.__name__ + message)
            except Exception as e:
                message = " setting mkl thread number didn't succeed"
                _logger.info(self.__class__.__name__ + message)

        evaluations = {}
        if os.path.isfile(os.path.join(self.cache_path, self.filename)):
            evaluations = pickle.load(
                open(os.path.join(self.cache_path, self.filename), 'rb'))

        already_evaluated = np.array([li in evaluations for li in self.labels])

        if not np.all(already_evaluated):
            samp = self.sampling.do_sampling()
        else:
            return list(evaluations.values())

        # setting random states
        for i in range(len(self.classifiers)):
            clf_params = self.classifiers[i].get_params()
            if 'random_state' in clf_params:
                clf_params['random_state'] = self.random_state
                self.classifiers[i] = self.classifiers[i].__class__(
                    **clf_params)
            if isinstance(self.classifiers[i], CalibratedClassifierCV):
                clf_params = self.classifiers[i].base_estimator.get_params()
                clf_params['random_state'] = self.random_state
                class_inst = self.classifiers[i].base_estimator.__class__
                new_inst = class_inst(**clf_params)
                self.classifiers[i].base_estimator = new_inst

        for i in range(len(self.classifiers)):
            if not already_evaluated[i]:
                message = " do the evaluation %s %s %s"
                message = message % (self.sampling.db_name,
                                     self.sampling.sampler.__name__,
                                     self.classifiers[i].__class__.__name__)
                _logger.info(self.__class__.__name__ + message)
                all_preds, all_tests, all_folds = [], [], []
                minority_class_label = None
                majority_class_label = None
                fold_idx = -1
                for X_train, y_train, X_test, y_test in samp['sampling']:
                    fold_idx += 1

                    # X_train[X_train == np.inf]= 0
                    # X_train[X_train == -np.inf]= 0
                    # X_test[X_test == np.inf]= 0
                    # X_test[X_test == -np.inf]= 0

                    class_labels = np.unique(y_train)
                    min_class_size = np.min(
                        [np.sum(y_train == c) for c in class_labels])

                    ss = StandardScaler()
                    X_train_trans = ss.fit_transform(X_train)
                    nonzero_var_idx = np.where(ss.var_ > 1e-8)[0]
                    X_test_trans = ss.transform(X_test)

                    enough_minority_samples = min_class_size > 4
                    y_train_big_enough = len(y_train) > 4
                    two_classes = len(class_labels) > 1
                    at_least_one_feature = (len(nonzero_var_idx) > 0)

                    if not enough_minority_samples:
                        message = " not enough minority samples: %d"
                        message = message % min_class_size
                        _logger.warning(
                            self.__class__.__name__ + message)
                    elif not y_train_big_enough:
                        message = (" number of minority training samples is "
                                   "not enough: %d")
                        message = message % len(y_train)
                        _logger.warning(self.__class__.__name__ + message)
                    elif not two_classes:
                        message = " there is only 1 class in training data"
                        _logger.warning(self.__class__.__name__ + message)
                    elif not at_least_one_feature:
                        _logger.warning(self.__class__.__name__ +
                                        (" no information in features"))
                    else:
                        all_tests.append(y_test)
                        if (minority_class_label is None or
                                majority_class_label is None):
                            class_labels = np.unique(y_train)
                            n_0 = sum(class_labels[0] == y_test)
                            n_1 = sum(class_labels[1] == y_test)
                            if n_0 < n_1:
                                minority_class_label = int(class_labels[0])
                                majority_class_label = int(class_labels[1])
                            else:
                                minority_class_label = int(class_labels[1])
                                majority_class_label = int(class_labels[0])

                        X_fit = X_train_trans[:, nonzero_var_idx]
                        self.classifiers[i].fit(X_fit, y_train)
                        clf = self.classifiers[i]
                        X_pred = X_test_trans[:, nonzero_var_idx]
                        pred = clf.predict_proba(X_pred)
                        all_preds.append(pred)
                        all_folds.append(
                            np.repeat(fold_idx, len(all_preds[-1])))

                if len(all_tests) > 0:
                    all_preds = np.vstack(all_preds)
                    all_tests = np.hstack(all_tests)
                    all_folds = np.hstack(all_folds)

                    evaluations[self.labels[i]] = self.calculate_metrics(
                        all_preds, all_tests, all_folds)
                else:
                    evaluations[self.labels[i]] = self.calculate_metrics(
                        None, None, None)

                evaluations[self.labels[i]]['runtime'] = samp['runtime']
                sampler_name = self.sampling.sampler.__name__
                evaluations[self.labels[i]]['sampler'] = sampler_name
                clf_name = self.classifiers[i].__class__.__name__
                evaluations[self.labels[i]]['classifier'] = clf_name
                sampler_parameters = self.sampling.sampler_parameters.copy()

                evaluations[self.labels[i]]['sampler_parameters'] = str(
                    sampler_parameters)
                evaluations[self.labels[i]]['classifier_parameters'] = str(
                    self.classifiers[i].get_params())
                evaluations[self.labels[i]]['sampler_categories'] = str(
                    self.sampling.sampler.categories)
                evaluations[self.labels[i]
                            ]['db_name'] = self.sampling.folding.db_name
                evaluations[self.labels[i]]['db_size'] = samp['db_size']
                evaluations[self.labels[i]]['db_n_attr'] = samp['db_n_attr']
                evaluations[self.labels[i]
                            ]['imbalanced_ratio'] = samp['imbalanced_ratio']

        if not np.all(already_evaluated):
            _logger.info(self.__class__.__name__ +
                         (" dumping to file %s" % self.filename))
            random_filename = os.path.join(self.cache_path, str(
                np.random.randint(1000000)) + '.pickle')
            pickle.dump(evaluations, open(random_filename, "wb"))
            os.rename(random_filename, os.path.join(
                self.cache_path, self.filename))

        return list(evaluations.values())


def trans(X):
    """
    Transformation function used to aggregate the evaluation results.

    Args:
        X (pd.DataFrame): a grouping of a data frame containing evaluation
                            results
    """
    auc_std = X.iloc[np.argmax(X['auc_mean'].values)]['auc_std']
    cp_auc = X.sort_values('auc')['classifier_parameters'].iloc[-1]
    cp_acc = X.sort_values('acc')['classifier_parameters'].iloc[-1]
    cp_gacc = X.sort_values('gacc')['classifier_parameters'].iloc[-1]
    cp_f1 = X.sort_values('f1')['classifier_parameters'].iloc[-1]
    cp_p_top20 = X.sort_values('p_top20')['classifier_parameters'].iloc[-1]
    cp_brier = X.sort_values('brier')['classifier_parameters'].iloc[-1]
    sp_auc = X.sort_values('auc')['sampler_parameters'].iloc[-1]
    sp_acc = X.sort_values('acc')['sampler_parameters'].iloc[-1]
    sp_gacc = X.sort_values('gacc')['sampler_parameters'].iloc[-1]
    sp_f1 = X.sort_values('f1')['sampler_parameters'].iloc[-1]
    sp_p_top20 = X.sort_values('p_top20')['sampler_parameters'].iloc[-1]
    sp_brier = X.sort_values('p_top20')['sampler_parameters'].iloc[0]

    return pd.DataFrame({'auc': np.max(X['auc']),
                         'auc_mean': np.max(X['auc_mean']),
                         'auc_std': auc_std,
                         'brier': np.min(X['brier']),
                         'acc': np.max(X['acc']),
                         'f1': np.max(X['f1']),
                         'p_top20': np.max(X['p_top20']),
                         'gacc': np.max(X['gacc']),
                         'runtime': np.mean(X['runtime']),
                         'db_size': X['db_size'].iloc[0],
                         'db_n_attr': X['db_n_attr'].iloc[0],
                         'imbalanced_ratio': X['imbalanced_ratio'].iloc[0],
                         'sampler_categories': X['sampler_categories'].iloc[0],
                         'classifier_parameters_auc': cp_auc,
                         'classifier_parameters_acc': cp_acc,
                         'classifier_parameters_gacc': cp_gacc,
                         'classifier_parameters_f1': cp_f1,
                         'classifier_parameters_p_top20': cp_p_top20,
                         'classifier_parameters_brier': cp_brier,
                         'sampler_parameters_auc': sp_auc,
                         'sampler_parameters_acc': sp_acc,
                         'sampler_parameters_gacc': sp_gacc,
                         'sampler_parameters_f1': sp_f1,
                         'sampler_parameters_p_top20': sp_p_top20,
                         'sampler_parameters_brier': sp_brier,
                         }, index=[0])


def _clone_classifiers(classifiers):
    """
    Clones a set of classifiers

    Args:
        classifiers (list): a list of classifier objects
    """
    results = []
    for c in classifiers:
        if isinstance(c, MLPClassifierWrapper):
            results.append(c.copy())
        else:
            results.append(clone(c))

    return results


def _cache_samplings(folding,
                     samplers,
                     scaler,
                     max_n_sampler_par_comb=35,
                     n_jobs=1,
                     random_state=None):
    """

    """
    _logger.info("create sampling objects, random_state: %s" %
                 str(random_state or ""))
    sampling_objs = []

    random_state_init = random_state
    random_state = np.random.RandomState(random_state_init)

    _logger.info("samplers: %s" % str(samplers))
    for s in samplers:
        sampling_par_comb = s.parameter_combinations()
        _logger.info(sampling_par_comb)
        domain = np.array(list(range(len(sampling_par_comb))))
        n_random = min([len(sampling_par_comb), max_n_sampler_par_comb])
        random_indices = random_state.choice(domain, n_random, replace=False)
        _logger.info("random_indices: %s" % random_indices)
        sampling_par_comb = [sampling_par_comb[i] for i in random_indices]
        _logger.info(sampling_par_comb)

        for spc in sampling_par_comb:
            sampling_objs.append(Sampling(folding,
                                          s,
                                          spc,
                                          scaler,
                                          random_state_init))

    # sorting sampling objects to optimize execution
    def key(x):
        if (isinstance(x.sampler, ADG) or isinstance(x.sampler, AMSCO) or
                isinstance(x.sampler, DSRBF)):
            if 'proportion' in x.sampler_parameters:
                return 30 + x.sampler_parameters['proportion']
            else:
                return 30
        elif 'proportion' in x.sampler_parameters:
            return x.sampler_parameters['proportion']
        elif OverSampling.cat_memetic in x.sampler.categories:
            return 20
        else:
            return 10

    sampling_objs = list(reversed(sorted(sampling_objs, key=key)))

    # executing sampling in parallel
    _logger.info("executing %d sampling in parallel" % len(sampling_objs))
    Parallel(n_jobs=n_jobs, batch_size=1)(delayed(s.cache_sampling)()
                                          for s in sampling_objs)

    return sampling_objs


def _cache_evaluations(sampling_objs,
                       classifiers,
                       n_jobs=1,
                       random_state=None):
    # create evaluation objects
    _logger.info("create classifier jobs")
    evaluation_objs = []

    num_threads = None if n_jobs is None or n_jobs == 1 else 1

    for s in sampling_objs:
        evaluation_objs.append(Evaluation(s, _clone_classifiers(
            classifiers), num_threads, random_state))

    _logger.info("executing %d evaluation jobs in parallel" %
                 (len(evaluation_objs)))
    # execute evaluation in parallel
    evals = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(e.do_evaluation)() for e in evaluation_objs)

    return evals


def _read_db_results(cache_path_db):
    results = []
    evaluation_files = glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))

    for f in evaluation_files:
        eval_results = pickle.load(open(f, 'rb'))
        results.append(list(eval_results.values()))

    return results


def read_oversampling_results(datasets, cache_path=None, all_results=False):
    """
    Reads the results of the evaluation

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        cache_path (str): path to a cache directory
        all_results (bool): True to return all results, False to return an
                                aggregation

    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is
                        False
    """

    results = []
    for dataset_spec in datasets:

        # loading dataset if needed and determining dataset name
        if not isinstance(dataset_spec, dict):
            dataset = dataset_spec()
        else:
            dataset = dataset_spec

        if 'name' in dataset:
            dataset_name = dataset['name']
        else:
            dataset_name = dataset_spec.__name__

        dataset['name'] = dataset_name

        # determining dataset specific cache path
        cache_path_db = os.path.join(cache_path, dataset_name)

        # reading the results
        res = _read_db_results(cache_path_db)

        # concatenating the results
        _logger.info("concatenating results")
        db_res = [pd.DataFrame(r) for r in res]
        db_res = pd.concat(db_res).reset_index(drop=True)

        _logger.info("aggregating the results")
        if all_results is False:
            db_res = db_res.groupby(by=['db_name', 'classifier', 'sampler'])
            db_res.apply(trans).reset_index().drop('level_3', axis=1)

        results.append(db_res)

    return pd.concat(results).reset_index(drop=True)


def evaluate_oversamplers(datasets,
                          samplers,
                          classifiers,
                          cache_path,
                          validator=RepeatedStratifiedKFold(
                              n_splits=5, n_repeats=3),
                          scaler=None,
                          all_results=False,
                          remove_cache=False,
                          max_samp_par_comb=35,
                          n_jobs=1,
                          random_state=None):
    """
    Evaluates oversampling techniques using various classifiers on various
        datasets

    Args:
        datasets (list): list of datasets and/or dataset loaders - a dataset
                            is a dict with 'data', 'target' and 'name' keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        validator (obj): validator object
        scaler (obj): scaler object
        all_results (bool): True to return all results, False to return an
                                aggregation
        remove_cache (bool): True to remove sampling objects after
                                        evaluation
        max_samp_par_comb (int): maximum number of sampler parameter
                                    combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        pd.DataFrame: all results or the aggregated results if all_results is
                        False

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= [imbd.load_glass2, imbd.load_ecoli4]
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]

        cache_path= '/home/<user>/smote_validation/'

        results= evaluate_oversamplers(datasets,
                                       oversamplers,
                                       classifiers,
                                       cache_path)
    """

    if cache_path is None:
        raise ValueError('cache_path is not specified')

    results = []
    for dataset_spec in datasets:
        # loading dataset if needed and determining dataset name
        if not isinstance(dataset_spec, dict):
            dataset = dataset_spec()
        else:
            dataset = dataset_spec

        if 'name' in dataset:
            dataset_name = dataset['name']
        else:
            dataset_name = dataset_spec.__name__

        dataset['name'] = dataset_name

        dataset_original_target = dataset['target'].copy()
        class_labels = np.unique(dataset['target'])
        n_0 = sum(dataset['target'] == class_labels[0])
        n_1 = sum(dataset['target'] == class_labels[1])
        if n_0 < n_1:
            min_label = class_labels[0]
            maj_label = class_labels[1]
        else:
            min_label = class_labels[1]
            maj_label = class_labels[0]
        min_ind = np.where(dataset['target'] == min_label)[0]
        maj_ind = np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)

        cache_path_db = os.path.join(cache_path, dataset_name)
        if not os.path.isdir(cache_path_db):
            _logger.info("creating cache directory")
            os.makedirs(cache_path_db)

        # checking of samplings and evaluations are available
        samplings_available = False
        evaluations_available = False

        samplings = glob.glob(os.path.join(cache_path_db, 'sampling*.pickle'))
        if len(samplings) > 0:
            samplings_available = True

        evaluations = glob.glob(os.path.join(cache_path_db, 'eval*.pickle'))
        if len(evaluations) > 0:
            evaluations_available = True

        message = ("dataset: %s, samplings_available: %s, "
                   "evaluations_available: %s")
        message = message % (dataset_name, str(samplings_available),
                             str(evaluations_available))
        _logger.info(message)

        if (remove_cache and evaluations_available and
                not samplings_available):
            # remove_cache is enabled and evaluations are available,
            # they are being read
            message = ("reading result from cache, sampling and evaluation is"
                       " not executed")
            _logger.info(message)
            res = _read_db_results(cache_path_db)
        else:
            _logger.info("doing the folding")
            folding = Folding(dataset, validator, cache_path_db, random_state)
            folding.do_folding()

            _logger.info("do the samplings")
            sampling_objs = _cache_samplings(folding,
                                             samplers,
                                             scaler,
                                             max_samp_par_comb,
                                             n_jobs,
                                             random_state)

            _logger.info("do the evaluations")
            res = _cache_evaluations(
                sampling_objs, classifiers, n_jobs, random_state)

        dataset['target'] = dataset_original_target

        # removing samplings once everything is done
        if remove_cache:
            filenames = glob.glob(os.path.join(cache_path_db, 'sampling*'))
            _logger.info("removing unnecessary sampling files")
            if len(filenames) > 0:
                for f in filenames:
                    os.remove(f)

        _logger.info("concatenating the results")
        db_res = [pd.DataFrame(r) for r in res]
        db_res = pd.concat(db_res).reset_index(drop=True)

        random_filename = os.path.join(cache_path_db, str(
            np.random.randint(1000000)) + '.pickle')
        pickle.dump(db_res, open(random_filename, "wb"))
        os.rename(random_filename, os.path.join(
            cache_path_db, 'results.pickle'))

        _logger.info("aggregating the results")
        if all_results is False:
            db_res = db_res.groupby(by=['db_name', 'classifier', 'sampler'])
            db_res = db_res.apply(trans).reset_index().drop('level_3', axis=1)

        results.append(db_res)

    return pd.concat(results).reset_index(drop=True)


def model_selection(dataset,
                    samplers,
                    classifiers,
                    cache_path,
                    score='auc',
                    validator=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
                    remove_cache=False,
                    max_samp_par_comb=35,
                    n_jobs=1,
                    random_state=None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best
    performance

    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name'
                        keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        cache_path (str): path to a cache directory
        score (str): 'auc'/'acc'/'gacc'/'f1'/'brier'/'p_top20'
        validator (obj): validator object
        all_results (bool): True to return all results, False to return an
                            aggregation
        remove_cache (bool): True to remove sampling objects after
                                        evaluation
        max_samp_par_comb (int): maximum number of sampler parameter
                                    combinations to be tested
        n_jobs (int): number of parallel jobs
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        obj, obj: the best performing sampler object and the best performing
                    classifier object

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        datasets= imbd.load_glass2()
        oversamplers= [sv.SMOTE_ENN, sv.NEATER, sv.Lee]
        classifiers= [KNeighborsClassifier(n_neighbors= 3),
                      KNeighborsClassifier(n_neighbors= 5),
                      DecisionTreeClassifier()]

        cache_path= '/home/<user>/smote_validation/'

        sampler, classifier= model_selection(dataset,
                                             oversamplers,
                                             classifiers,
                                             cache_path,
                                             'auc')
    """

    if score not in ['auc', 'acc', 'gacc', 'f1', 'brier', 'p_top20']:
        raise ValueError("score %s not supported" % score)

    results = evaluate_oversamplers(datasets=[dataset],
                                    samplers=samplers,
                                    classifiers=classifiers,
                                    cache_path=cache_path,
                                    validator=validator,
                                    remove_cache=remove_cache,
                                    max_samp_par_comb=max_samp_par_comb,
                                    n_jobs=n_jobs,
                                    random_state=random_state)

    # extracting the best performing classifier and oversampler parameters
    # regarding AUC
    highest_score = results[score].idxmax()
    cl_par_name = 'classifier_parameters_' + score
    samp_par_name = 'sampler_parameters_' + score
    cl, cl_par, samp, samp_par = results.loc[highest_score][['classifier',
                                                             cl_par_name,
                                                             'sampler',
                                                             samp_par_name]]

    # instantiating the best performing oversampler and classifier objects
    samp_obj = eval(samp)(**eval(samp_par))
    cl_obj = eval(cl)(**eval(cl_par))

    return samp_obj, cl_obj


def cross_validate(dataset,
                   sampler,
                   classifier,
                   validator=RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
                   scaler=StandardScaler(),
                   random_state=None):
    """
    Evaluates oversampling techniques on various classifiers and a dataset
    and returns the oversampling and classifier objects giving the best
    performance

    Args:
        dataset (dict): a dataset is a dict with 'data', 'target' and 'name'
                        keys
        samplers (list): list of oversampling classes/objects
        classifiers (list): list of classifier objects
        validator (obj): validator object
        scaler (obj): scaler object
        random_state (int/np.random.RandomState/None): initializer of the
                                                        random state

    Returns:
        pd.DataFrame: the cross-validation scores

    Example::

        import smote_variants as sv
        import imbalanced_datasets as imbd

        from sklearn.neighbors import KNeighborsClassifier

        dataset= imbd.load_glass2()
        sampler= sv.SMOTE_ENN
        classifier= KNeighborsClassifier(n_neighbors= 3)

        sampler, classifier= model_selection(dataset,
                                             oversampler,
                                             classifier)
    """

    class_labels = np.unique(dataset['target'])
    binary_problem = (len(class_labels) == 2)

    dataset_orig_target = dataset['target'].copy()
    if binary_problem:
        _logger.info("The problem is binary")
        n_0 = sum(dataset['target'] == class_labels[0])
        n_1 = sum(dataset['target'] == class_labels[1])
        if n_0 < n_1:
            min_label = class_labels[0]
            maj_label = class_labels[1]
        else:
            min_label = class_labels[0]
            maj_label = class_labels[1]

        min_ind = np.where(dataset['target'] == min_label)[0]
        maj_ind = np.where(dataset['target'] == maj_label)[0]
        np.put(dataset['target'], min_ind, 1)
        np.put(dataset['target'], maj_ind, 0)
    else:
        _logger.info("The problem is not binary")
        label_indices = {}
        for c in class_labels:
            label_indices[c] = np.where(dataset['target'] == c)[0]
        mapping = {}
        for i, c in enumerate(class_labels):
            np.put(dataset['target'], label_indices[c], i)
            mapping[i] = c

    runtimes = []
    all_preds, all_tests = [], []

    for train, test in validator.split(dataset['data'], dataset['target']):
        _logger.info("Executing fold")
        X_train, y_train = dataset['data'][train], dataset['target'][train]
        X_test, y_test = dataset['data'][test], dataset['target'][test]

        begin = time.time()
        X_samp, y_samp = sampler.sample(X_train, y_train)
        runtimes.append(time.time() - begin)

        X_samp_trans = scaler.fit_transform(X_samp)
        nonzero_var_idx = np.where(scaler.var_ > 1e-8)[0]
        X_test_trans = scaler.transform(X_test)

        all_tests.append(y_test)

        classifier.fit(X_samp_trans[:, nonzero_var_idx], y_samp)
        all_preds.append(classifier.predict_proba(
            X_test_trans[:, nonzero_var_idx]))

    if len(all_tests) > 0:
        all_preds = np.vstack(all_preds)
        all_tests = np.hstack(all_tests)

    dataset['target'] = dataset_orig_target

    _logger.info("Computing the results")

    results = {}
    results['runtime'] = np.mean(runtimes)
    results['sampler'] = sampler.__class__.__name__
    results['classifier'] = classifier.__class__.__name__
    results['sampler_parameters'] = str(sampler.get_params())
    results['classifier_parameters'] = str(classifier.get_params())
    results['db_size'] = len(dataset['data'])
    results['db_n_attr'] = len(dataset['data'][0])
    results['db_n_classes'] = len(class_labels)

    if binary_problem:
        results['imbalance_ratio'] = sum(
            dataset['target'] == maj_label)/sum(dataset['target'] == min_label)
        all_pred_labels = np.apply_along_axis(
            lambda x: np.argmax(x), 1, all_preds)

        results['tp'] = np.sum(np.logical_and(
            np.equal(all_tests, all_pred_labels), (all_tests == 1)))
        results['tn'] = np.sum(np.logical_and(
            np.equal(all_tests, all_pred_labels), (all_tests == 0)))
        results['fp'] = np.sum(np.logical_and(np.logical_not(
            np.equal(all_tests, all_pred_labels)), (all_tests == 0)))
        results['fn'] = np.sum(np.logical_and(np.logical_not(
            np.equal(all_tests, all_pred_labels)), (all_tests == 1)))
        results['p'] = results['tp'] + results['fn']
        results['n'] = results['fp'] + results['tn']
        results['acc'] = (results['tp'] + results['tn']) / \
            (results['p'] + results['n'])
        results['sens'] = results['tp']/results['p']
        results['spec'] = results['tn']/results['n']
        results['ppv'] = results['tp']/(results['tp'] + results['fp'])
        results['npv'] = results['tn']/(results['tn'] + results['fn'])
        results['fpr'] = 1.0 - results['spec']
        results['fdr'] = 1.0 - results['ppv']
        results['fnr'] = 1.0 - results['sens']
        results['bacc'] = (results['tp']/results['p'] +
                           results['tn']/results['n'])/2.0
        results['gacc'] = np.sqrt(
            results['tp']/results['p']*results['tn']/results['n'])
        results['f1'] = 2*results['tp'] / \
            (2*results['tp'] + results['fp'] + results['fn'])
        mcc_num = (results['tp']*results['tn'] - results['fp']*results['fn'])
        tp_fp = (results['tp'] + results['fp'])
        tp_fn = (results['tp'] + results['fn'])
        tn_fp = (results['tn'] + results['fp'])
        tn_fn = (results['tn'] + results['fn'])
        mcc_denom = np.sqrt(tp_fp * tp_fn * tn_fp * tn_fn)
        results['mcc'] = mcc_num/mcc_denom
        results['l'] = (results['p'] + results['n']) * \
            np.log(results['p'] + results['n'])
        results['ltp'] = results['tp']*np.log(results['tp']/(
            (results['tp'] + results['fp'])*(results['tp'] + results['fn'])))
        results['lfp'] = results['fp']*np.log(results['fp']/(
            (results['fp'] + results['tp'])*(results['fp'] + results['tn'])))
        results['lfn'] = results['fn']*np.log(results['fn']/(
            (results['fn'] + results['tp'])*(results['fn'] + results['tn'])))
        results['ltn'] = results['tn']*np.log(results['tn']/(
            (results['tn'] + results['fp'])*(results['tn'] + results['fn'])))
        results['lp'] = results['p'] * \
            np.log(results['p']/(results['p'] + results['n']))
        results['ln'] = results['n'] * \
            np.log(results['n']/(results['p'] + results['n']))
        ucc_num = (results['l'] + results['ltp'] + results['lfp'] +
                   results['lfn'] + results['ltn'])
        results['uc'] = ucc_num/(results['l'] + results['lp'] + results['ln'])
        results['informedness'] = results['sens'] + results['spec'] - 1.0
        results['markedness'] = results['ppv'] + results['npv'] - 1.0
        results['log_loss'] = log_loss(all_tests, all_preds)
        results['auc'] = roc_auc_score(all_tests, all_preds[:, 1])
        test_labels, preds = zip(
            *sorted(zip(all_tests, all_preds[:, 1]), key=lambda x: -x[1]))
        test_labels = np.array(test_labels)
        th = int(0.2*len(test_labels))
        results['p_top20'] = np.sum(test_labels[:th] == 1)/th
        results['brier'] = np.mean((all_preds[:, 1] - all_tests)**2)
    else:
        all_pred_labels = np.apply_along_axis(
            lambda x: np.argmax(x), 1, all_preds)

        results['acc'] = accuracy_score(all_tests, all_pred_labels)
        results['confusion_matrix'] = confusion_matrix(
            all_tests, all_pred_labels)
        sum_confusion = np.sum(results['confusion_matrix'], axis=0)
        results['gacc'] = gmean(np.diagonal(
            results['confusion_matrix'])/sum_confusion)
        results['class_label_mapping'] = mapping

    return pd.DataFrame({'value': list(results.values())},
                        index=results.keys())
