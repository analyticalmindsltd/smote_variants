"""
This module implements the SMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['SMOTE']

class SMOTE(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_metric_learning]

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
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex samplign parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        OverSamplingSimplex.__init__(self, **ss_params, random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params= coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @classmethod
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

        X_min = X[y == self.min_label]

        # fitting the model
        n_neighbors = min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn_mt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nn_mt.fit(X_min)
        _, ind_min = nn_mt.kneighbors(X_min, return_distance=True)

        samples = self.sample_simplex(X=X_min,
                                        indices=ind_min,
                                        n_to_sample=n_to_sample)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self, deep)}
