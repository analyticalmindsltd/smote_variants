"""
This module implements the SL_graph_SMOTE method.
"""

import numpy as np

from scipy.stats import skew

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from ._safe_level_smote import Safe_Level_SMOTE
from ._borderline_smote import Borderline_SMOTE1

from .._logger import logger
_logger= logger

__all__= ['SL_graph_SMOTE']

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
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
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
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, ss_params_default)
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
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
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # Fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        indices = nnmt.kneighbors(X[y == self.min_label], return_distance=False)

        # Computing safe level values
        safe_level_values = np.sum(y[indices], axis=1)

        # Computing skewness
        skewness = skew(safe_level_values)

        if skewness < 0:
            # left skewed
            sampler = Safe_Level_SMOTE(proportion=self.proportion,
                                        n_neighbors=self.n_neighbors,
                                        nn_params=nn_params,
                                        ss_params=self.ss_params,
                                        n_jobs=self.n_jobs,
                                        random_state=self._random_state_init)
        else:
            # right skewed
            sampler = Borderline_SMOTE1(proportion=self.proportion,
                                        n_neighbors=self.n_neighbors,
                                        nn_params=nn_params,
                                        ss_params=self.ss_params,
                                        n_jobs=self.n_jobs,
                                        random_state=self._random_state_init)

        return sampler.sample(X, y)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
