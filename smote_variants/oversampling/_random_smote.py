"""
This module implements the Random_SMOTE method.
"""

import numpy as np

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['Random_SMOTE']

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
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
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
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn

        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
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
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model to find closest neighbors of minority
        # points
        n_neighbors = min([len(X_min), self.n_neighbors + 1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        base_indices = self.random_state.choice(np.arange(X_min.shape[0]),
                                                n_to_sample)
        y12_ind = self.random_state.choice(np.arange(1, n_neighbors),
                                            size=(n_to_sample, 2))

        y1_ind = X_min[ind[base_indices, y12_ind[:, 0]]]
        y2_ind = X_min[ind[base_indices, y12_ind[:, 1]]]

        tmp = y1_ind + (y2_ind - y1_ind) \
                        * self.random_state.random_sample(size=y1_ind.shape)

        samples = X_min[base_indices] + (tmp - X_min[base_indices]) \
                        * self.random_state.random_sample(size=tmp.shape)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
