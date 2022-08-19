"""
This module implements the SMOTE_ENN method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import OverSampling
from ..noise_removal import EditedNearestNeighbors
from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_ENN']

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
                  OverSampling.cat_changes_majority,
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
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
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
        return SMOTE.parameter_combinations(raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        smote = SMOTE(proportion=self.proportion,
                      n_neighbors=self.n_neighbors,
                      nn_params=nn_params,
                      ss_params=self.ss_params,
                      n_jobs=self.n_jobs,
                      random_state=self._random_state_init)
        X_new, y_new = smote.sample(X, y)

        enn = EditedNearestNeighbors(n_jobs=self.n_jobs,
                                     nn_params=nn_params)

        X_res, y_res = enn.remove_noise(X_new, y_new) # pylint: disable=invalid-name

        if (np.sum(y_res == self.maj_label) < np.sum(y_res == self.min_label))\
            or np.unique(y_res).shape[0] < 2:
            return self.return_copies(X, y,
                                "ENN removed too many samples")
        return X_res, y_res

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
