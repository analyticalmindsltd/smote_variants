"""
This module implements the Borderline_SMOTE methods.
"""

import numpy as np

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from ..base import mode, coalesce_dict, coalesce

from .._logger import logger
_logger= logger

__all__= ['Borderline_SMOTE1',
          'Borderline_SMOTE2']

def determine_danger_remove_noise(*, X, y, X_min,
                                    nn_params,
                                    n_neighbors,
                                    n_jobs,
                                    maj_label):
    """
    Determine in-danger items and remove noise.

    Args:
        X (np.array): all samples
        y (np.array): all target values
        X_min (np.array): all minority samples
        nn_params (dict): the nearest neighbors parameters
        n_neighbors (int): number of neighbors
        n_jobs (int): number of jobs
        maj_label (int): majority label

    Returns:
        np.array, np.array: X_min, X_danger
    """
    n_neighbors = np.min([len(X), n_neighbors + 1])

    nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                            n_jobs=n_jobs,
                                            **(nn_params))
    nnmt.fit(X)
    indices= nnmt.kneighbors(X_min, return_distance=False)

    # determining minority samples in danger
    noise = []
    danger = []
    for idx, row in enumerate(indices):
        if (n_neighbors - 1) == sum(y[row[1:]] == maj_label):
            noise.append(idx)
        elif mode(y[row[1:]]) == maj_label:
            danger.append(idx)

    X_danger = X_min[danger] # pylint: disable=invalid-name
    X_min = np.delete(X_min, np.array(noise).astype(int), axis=0)

    return X_min, X_danger

class Borderline_SMOTE1(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling Method
                                     in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
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
            n_neighbors (int): control parameter of the nearest neighbor
                                    technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                    technique for sampling
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
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
        self.nn_params= coalesce(nn_params, {})
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
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}

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

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # fitting model
        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params,
                                                                        X, y)

        X_min, X_danger = determine_danger_remove_noise(X=X, y=y, # pylint: disable=invalid-name
                                                        X_min=X_min,
                                                        nn_params=nn_params,
                                                        n_neighbors=self.n_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        maj_label=self.maj_label)

        if len(X_min) < 2 or len(X_danger) == 0:
            return self.return_copies(X, y, "The number of samples "\
                    f"after preprocessing X_min ({len(X_min)}) "\
                    f" X_danger ({len(X_danger)} is not enough for sampling.")

        # fitting nearest neighbors model to minority samples
        k_neigh = min([len(X_min), self.k_neighbors + 1])

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=k_neigh,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        indices= nnmt.kneighbors(X_danger, return_distance=False)

        # X_danger items are also present in X_min
        indices = indices[:, 1:]

        samples = self.sample_simplex(X=X_danger,
                                        indices=indices,
                                        n_to_sample=n_to_sample,
                                        X_vertices=X_min)

        # original implementation
        # generating samples near points in danger
        # base_indices = self.random_state.choice(list(range(len(X_danger))),
        #                                        n_to_sample)
        #distribution = np.repeat(1, k_neigh-1)
        #distribution = distribution / np.sum(distribution)
        #
        #neighbor_indices = self.random_state.choice(list(range(1, k_neigh)),
        #                                            n_to_sample,
        #                                            p=distribution)
        #
        #X_base = X_danger[base_indices]
        #X_neighbor = X_min[indices[base_indices, neighbor_indices]]
        #
        #samples = X_base + \
        #    np.multiply(self.random_state.rand(
        #        n_to_sample, 1), X_neighbor - X_base)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}


class Borderline_SMOTE2(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{borderlineSMOTE,
                            author="Han, Hui
                            and Wang, Wen-Yuan
                            and Mao, Bing-Huan",
                            editor="Huang, De-Shuang
                            and Zhang, Xiao-Ping
                            and Huang, Guang-Bin",
                            title="Borderline-SMOTE: A New Over-Sampling
                                    Method in Imbalanced Data Sets Learning",
                            booktitle="Advances in Intelligent Computing",
                            year="2005",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="878--887",
                            isbn="978-3-540-31902-3"
                            }
    """

    categories = [OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 k_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique for determining the borderline
            k_neighbors (int): control parameter of the nearest neighbor
                                technique for sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            preferential (bool): whether to use preferential connectivity in the
                                    neighborhoods
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(k_neighbors, 'k_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.k_neighbors = k_neighbors
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
                                  'n_neighbors': [3, 5, 7],
                                  'k_neighbors': [3, 5, 7]}
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

        # determining number of samples to be generated
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model
        X_min = X[y == self.min_label]

        X_min, X_danger = determine_danger_remove_noise(X=X, y=y, # pylint: disable=invalid-name
                                                        X_min=X_min,
                                                        nn_params=nn_params,
                                                        n_neighbors=self.n_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        maj_label=self.maj_label)

        if len(X_min) < 2 or len(X_danger) == 0:
            return self.return_copies(X, y, "The number of samples "\
                    f"after preprocessing X_min ({len(X_min)}) "\
                    f" X_danger ({len(X_danger)} is not enough for sampling.")

        # fitting nearest neighbors model to samples
        k_neigh = self.k_neighbors + 1
        k_neigh = min([k_neigh, len(X)])

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=k_neigh,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        indices= nnmt.kneighbors(X_danger, return_distance=False)

        # X_danger items are also present in X
        indices = indices[:, 1:]

        vertex_weights = np.repeat(1.0, len(X))
        vertex_weights[y == self.maj_label] = 0.5

        samples = self.sample_simplex(X=X_danger,
                                        indices=indices,
                                        n_to_sample=n_to_sample,
                                        X_vertices=X,
                                        vertex_weights=vertex_weights)

        # generating the samples
        #base_indices = self.random_state.choice(
        #    list(range(len(X_danger))), n_to_sample)
        #
        #distribution = np.repeat(1, k_neigh-1)
        #distribution = distribution / np.sum(distribution)
        #
        #neighbor_indices = self.random_state.choice(
        #    list(range(1, k_neigh)), n_to_sample, p=distribution)
        #
        #X_base = X_danger[base_indices]
        #X_neighbor = X[indices[base_indices, neighbor_indices]]
        #diff = X_neighbor - X_base
        #rand = self.random_state.rand(n_to_sample, 1)
        #mask = y[neighbor_indices] == self.maj_label
        #rand[mask] = rand[mask]*0.5
        #
        #samples = X_base + np.multiply(rand, diff)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'k_neighbors': self.k_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
