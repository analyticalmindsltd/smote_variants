"""
This module implements the ISMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['ISMOTE']

class ISMOTE(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_changes_majority,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 minority_weight=0.5,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            minority_weight (float): weight parameter according to the paper
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        nn_params = coalesce(nn_params, {})
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(minority_weight, "minority_weight", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.minority_weight = minority_weight
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'minority_weight': [0.2, 0.5, 0.8]}
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
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_to_sample = int((len(X_maj) - len(X_min))/2 + 0.5)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # computing distances of majority samples from minority ones
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=len(X_min),
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        dist, ind = nnmt.kneighbors(X_maj)

        # sort majority instances in descending order by their mean distance
        # from minority samples
        to_sort = zip(np.arange(len(X_maj)), np.mean(dist, axis=1))
        ind_sorted, _ = zip(*sorted(to_sort, key=lambda x: -x[1]))

        # remove the ones being farthest from the minority samples
        X_maj = X_maj[list(ind_sorted[n_to_sample:])]

        # construct new dataset
        X = np.vstack([X_maj, X_min])
        y = np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min))])

        # fitting nearest neighbors model
        n_neighbors = np.min([len(X), self.n_neighbors + 1])
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        vertex_weights = np.repeat(1.0 - self.minority_weight, X.shape[0])
        vertex_weights[y == self.min_label] = self.minority_weight

        # removing the minority samples from their own neighborhoods
        ind = ind[:, 1:]

        samples = self.sample_simplex(X=X_min,
                                        indices=ind,
                                        n_to_sample=n_to_sample,
                                        X_vertices=X,
                                        vertex_weights=vertex_weights)

        # do the oversampling
        #samples = []
        #while len(samples) < n_to_sample:
        #    idx = self.random_state.choice(np.arange(len(X_min)))
        #    y_idx = self.random_state.choice(ind[idx][1:])

            # different generation scheme depending on the class label
        #    if y_new[y_idx] == self.min_label:
        #        diff = (X_new[y_idx] - X_min[idx])
        #        r = self.random_state.random_sample()
        #        samples.append(X_min[idx] + r * diff * self.minority_weight)
        #    else:
        #        diff = (X_new[y_idx] - X_min[idx])
        #        r = self.random_state.random_sample()
        #        sample = X_min[idx] + r * diff * (1.0 - self.minority_weight)
        #        samples.append(sample)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'minority_weight': self.minority_weight,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
