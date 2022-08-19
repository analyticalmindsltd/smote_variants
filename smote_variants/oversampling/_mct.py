"""
This module implements the MCT method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict, fix_density
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['MCT']

class MCT(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_copy]

    def __init__(self,
                 *,
                 proportion=1.0,
                 n_neighbors=5,
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): additional parameter added by the developers of
                                smote-variants to support simplex sampling
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
        ss_params_default = {'n_dim': 1, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
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
                                  'n_neighbors': [5]}
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

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        n_neigh = min([len(X_min), self.n_neighbors+1])
        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neigh,
                                                   n_jobs=self.n_jobs,
                                                   **nn_params)
        nearestn.fit(X_min)
        indices = nearestn.kneighbors(X_min, return_distance=False)

        # having continuous variables, the mode is replaced by median
        x_med = np.median(X_min, axis=0)
        distances = np.linalg.norm(x_med - X_min, axis=1)
        distribution = fix_density(np.sum(distances) - distances)

        # do the sampling
        samples = self.sample_simplex(X=X_min,
                                        indices=indices,
                                        n_to_sample=n_to_sample,
                                        base_weights=distribution)
        #samples = []
        #while len(samples) < n_to_sample:
        #    samples.append(X_min[self.random_state.choice(
        #        np.arange(len(X_min)), p=distribution)])

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
                **OverSamplingSimplex.get_params(self)}
