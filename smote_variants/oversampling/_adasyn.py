"""
This module implements the ADASYN technique.
"""

import numpy as np

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from ..base import coalesce, coalesce_dict

from .._logger import logger
_logger= logger

__all__= ['ADASYN']

class ADASYN(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_density_based,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 beta=1.0,
                 *,
                 d_th=0.9,
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 proportion=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            d_th (float): tolerated deviation level from balancedness
            beta (float): target level of balancedness, same as proportion
                            in other techniques
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling params
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
            proportion (float): same as beta, for convenience
        """
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state, checks=None)

        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(d_th, 'd_th', 0)
        self.check_greater_or_equal(beta, 'beta', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.d_th = d_th
        self.beta = proportion or beta
        self.proportion = self.beta
        self.nn_params = nn_params
        self.n_jobs = n_jobs

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7, 9],
                                  'd_th': [0.9],
                                  'proportion': [2.0, 1.5, 1.0, 0.75, 0.5, 0.25]}
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
        n_to_sample = self.det_n_to_sample(self.beta)

        if n_to_sample == 0:
            return self.return_copies(X, y, "no need for sampling")

        # extracting minority samples
        X_min = X[y == self.min_label]

        # checking if sampling is needed
        m_min = len(X_min)
        m_maj = len(X) - m_min

        d = float(m_min)/m_maj # pylint: disable=invalid-name
        if d > self.d_th: # pylint: disable=invalid-name
            _logger.warning("%s: d > d_th",
                            self.__class__.__name__)
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model to all samples
        n_neigh = min([len(X_min), self.n_neighbors+1])
        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neigh,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nearestn.fit(X)
        indices = nearestn.kneighbors(X_min, return_distance=False)

        # determining the distribution of points to be generated
        r = [] #pylint: disable=invalid-name
        for _, row in enumerate(indices):
            r.append(sum(y[row[1:]] ==
                         self.maj_label)/self.n_neighbors)

        r = np.array(r) #pylint: disable=invalid-name

        if np.sum(r) == 0:
            _logger.warning("%s: not enough samples close to majority ones "\
                            "for oversampling",
                            self.__class__.__name__)
            return X.copy(), y.copy()

        r = r/np.sum(r) #pylint: disable=invalid-name

        # fitting nearest neighbors models to minority samples
        n_neigh = min([len(X_min), self.n_neighbors + 1])
        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neigh,
                                                        n_jobs=self.n_jobs,
                                                        **nn_params)
        nearestn.fit(X_min)
        indices = nearestn.kneighbors(X_min, return_distance=False)

        samples = self.sample_simplex(X=X_min,
                                        indices=indices,
                                        n_to_sample=n_to_sample,
                                        base_weights=r)

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
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'proportion': self.proportion,
                **OverSamplingSimplex.get_params(self)}
