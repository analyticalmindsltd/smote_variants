"""
This module implements the ISOMAP_Hybrid method.
"""
import warnings

import numpy as np

from sklearn.manifold import Isomap

from ..config import suppress_external_warnings
from ..base import coalesce, coalesce_dict
from ..base import OverSampling
from ._smote import SMOTE
from ..noise_removal import NeighborhoodCleaningRule

from .._logger import logger
_logger= logger

__all__= ['ISOMAP_Hybrid']

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
                  OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 n_components=3,
                 smote_n_neighbors=5,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_components (int): number of components
            smote_n_neighbors (int): number of neighbors in SMOTE sampling
            n_jobs (int): number of parallel jobs
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_components, "n_components", 1)
        self.check_greater_or_equal(smote_n_neighbors, "smote_n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.isomap_params = {}
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, ss_params_default)
        self.smote_n_neighbors = smote_n_neighbors
        self.n_jobs = n_jobs

        self.isomap = Isomap(n_neighbors=n_neighbors,
                             n_components=n_components,
                             n_jobs=n_jobs)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_components': [2, 3, 4],
                                  'smote_n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        with warnings.catch_warnings():
            if suppress_external_warnings():
                warnings.simplefilter("ignore")
            X_trans = self.isomap.fit_transform(X, y) # pylint: disable=invalid-name

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                    self.metric_tensor_from_nn_params(nn_params, X_trans, y)

        X_sm, y_sm = SMOTE(proportion=self.proportion, # pylint: disable=invalid-name
                           n_neighbors=self.smote_n_neighbors,
                           nn_params=nn_params,
                           ss_params=self.ss_params,
                           n_jobs=self.n_jobs,
                           random_state=self._random_state_init).sample(X_trans, y)

        ncr = NeighborhoodCleaningRule(n_jobs=self.n_jobs,
                                      nn_params=nn_params)
        X_final, y_final= ncr.remove_noise(X_sm, y_sm)

        if np.sum(y_final) <= 2 or np.sum(y_final == 0) <= 2:
            return self.return_copies(X_sm, y_sm, "too many samples removed "\
                                    "returning isomapped dataset")

        return X_final, y_final

    def preprocessing_transform(self, X):
        """
        Transforms new data by the trained isomap

        Args:
            X (np.array): new data

        Returns:
            np.array: the transformed data
        """
        return self.isomap.transform(X)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.isomap.n_neighbors,
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'n_components': self.isomap.n_components,
                'smote_n_neighbors': self.smote_n_neighbors,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
