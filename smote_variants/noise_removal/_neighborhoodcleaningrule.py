"""
This module implements the neighborhood cleaning rule.
"""

import numpy as np

from ..base import mode, coalesce
from ._noisefilter import NoiseFilter
from ..base import (NearestNeighborsWithMetricTensor)

from .._logger import logger
_logger= logger

__all__= ['NeighborhoodCleaningRule']

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

    def __init__(self, nn_params=None, n_jobs=1, **_kwargs):
        """
        Constructor of the noise removal object

        Args:
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    def get_params(self, deep=False):
        return {'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **NoiseFilter.get_params(self, deep)}

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.array): features
            y (np.array): target labels

        Returns:
            np.array, np.array: cleaned features and target labels
        """
        _logger.info("%s: Running noise removal", self.__class__.__name__)
        self.class_label_statistics(y)

        # fitting nearest neighbors with proposed parameter
        # using 4 neighbors because the first neighbor is the point itself
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=4,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        indices = nnmt.kneighbors(X, return_distance=False)

        # identifying the samples to be removed
        to_remove = []
        for idx in range(len(X)):
            if (y[idx] == self.maj_label and
                    mode(y[indices[idx][1:]]) == self.min_label):
                # if sample i is majority and the decision based on
                # neighbors is minority
                to_remove.append(idx)
            elif (y[idx] == self.min_label and
                  mode(y[indices[idx][1:]]) == self.maj_label):
                # if sample i is minority and the decision based on
                # neighbors is majority
                for jdx in indices[idx][1:]:
                    if y[jdx] == self.maj_label:
                        to_remove.append(jdx)

        # removing the noisy samples and returning the results
        to_remove = list(set(to_remove))
        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)
