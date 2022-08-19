"""
This module implements the Tomek-link removal technique.
"""

import numpy as np

from ._noisefilter import NoiseFilter
from ..base import (NearestNeighborsWithMetricTensor, coalesce)

from .._logger import logger
_logger= logger

__all__= ['TomekLinkRemoval']

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

    def __init__(self,
                 strategy='remove_majority',
                 nn_params=None,
                 n_jobs=1,
                 **_kwargs):
        """
        Constructor of the noise filter.

        Args:
            strategy (str): noise removal strategy:
                            'remove_majority'/'remove_both'
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_isin(strategy, 'strategy', ['remove_majority', 'remove_both'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.nn_params= coalesce(nn_params, {})
        self.n_jobs = n_jobs

    def get_params(self, deep=False):
        return {'strategy': self.strategy,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **NoiseFilter.get_params(self, deep)}

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.array): features
            y (np.array): target labels

        Returns:
            np.array, np.array: dataset after noise removal
        """
        _logger.info("%s: Running noise removal.", self.__class__.__name__)
        self.class_label_statistics(y)

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # using 2 neighbors because the first neighbor is the point itself
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=2,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        indices= nnmt.fit(X).kneighbors(X, return_distance=False)

        # identify links
        links = []
        for idx, row in enumerate(indices):
            if indices[row[1]][1] == idx:
                if not y[row[1]] == y[indices[row[1]][1]]:
                    links.append((idx, row[1]))

        # determine links to be removed
        to_remove = []
        for link in links:
            if self.strategy == 'remove_majority':
                if y[link[0]] == self.min_label:
                    to_remove.append(link[1])
                else:
                    to_remove.append(link[0])
            elif self.strategy == 'remove_both':
                to_remove.append(link[0])
                to_remove.append(link[1])

        to_remove = list(set(to_remove))

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)
