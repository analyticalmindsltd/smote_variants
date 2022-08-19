"""
This module implements the AND_SMOTE method.
"""

import numpy as np

from ..base import coalesce_dict, coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['AND_SMOTE']

class AND_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @inproceedings{and_smote,
                             author = {Yun, Jaesub and Ha,
                                 Jihyun and Lee, Jong-Seok},
                             title = {Automatic Determination of Neighborhood
                                        Size in SMOTE},
                             booktitle = {Proceedings of the 10th International
                                            Conference on Ubiquitous
                                            Information Management and
                                            Communication},
                             series = {IMCOM '16},
                             year = {2016},
                             isbn = {978-1-4503-4142-4},
                             location = {Danang, Viet Nam},
                             pages = {100:1--100:8},
                             articleno = {100},
                             numpages = {8},
                             doi = {10.1145/2857546.2857648},
                             acmid = {2857648},
                             publisher = {ACM},
                             address = {New York, NY, USA},
                             keywords = {SMOTE, imbalanced learning, synthetic
                                            data generation},
                            }
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 K=15,
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
            K (int): maximum number of nearest neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): the simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(K, "K", 2)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.K = K # pylint: disable=invalid-name
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
                                  'K': [9, 15, 21]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def calculate_r_min_maj(self, *, min_idx, neighbor_indices, neigh_idx, X, y):
        """
        Calculate the regions.

        Args:
            min_idx (int): minority index
            neighbor_indices (np.array): the neighbor indices
            neigh_idx (int): one neighbor index
            X (np.array): all features
            y (np.array): all targets

        Returns:
            set, set: the minority and majority regions
        """
        # region coordinates
        reg = np.hstack([min_idx, neighbor_indices[neigh_idx]])

        # all the points in the region must be among the neighbors
        # what we do is counting how many of them are minority and
        # majority samples
        reg_indices= neighbor_indices[:(neigh_idx+1)]
        X_all = X[neighbor_indices[:(neigh_idx+1)]]

        # comparison to the corener points, the mask determines whether
        # the elements of X_all are within the region
        mask = np.logical_and(np.all(np.min(X[reg], axis=0) <= X_all, axis=1),
                             np.all(X_all <= np.max(X[reg], axis=0), axis=1))

        min_mask = np.logical_and(mask, y[reg_indices] == self.min_label)
        maj_mask = np.logical_and(mask, y[reg_indices] == self.maj_label)

        r_min = set(reg_indices[min_mask].tolist())
        r_maj = set(reg_indices[maj_mask].tolist())

        return r_min, r_maj

    def calculate_kappa(self, min_idx, neighbor_indices, X, y):
        """
        Calculates the kappa measure as specified in the paper.

        Args:
            min_idx (int): the index of a minority sample
            neighbor_indices (np.array): the indices of its neighbors
            X (np.array): all features
            y (np.array): all targets

        Returns:
            int: the kappa measure
        """
        regions_min = []
        regions_maj = []

        for neigh_idx in range(1, neighbor_indices.shape[0]):
            # continuing if the label of the neighbors is not minority
            if y[neighbor_indices[neigh_idx]] == self.maj_label:
                continue

            r_min, r_maj = self.calculate_r_min_maj(min_idx=min_idx,
                                            neighbor_indices=neighbor_indices,
                                            neigh_idx=neigh_idx,
                                            X=X,
                                            y=y)

            # appending the coordinates of points to the minority and
            # majority regions
            if len(regions_min) == 0:
                regions_min.append(r_min)
                regions_maj.append(r_maj)
            else:
                regions_min.append(regions_min[-1].union(r_min))
                regions_maj.append(regions_maj[-1].union(r_maj))

        # computing the lengths of the increasing minority and majority
        # sets
        regions_min = np.array([len(r) for r in regions_min])
        regions_maj = np.array([len(r) for r in regions_maj])

        # computing the precision of minority classification (all points
        # are supposed to be classified as minority)
        prec = regions_min/(regions_min + regions_maj)
        # discrete differentiation with order 1
        discrete_difference = np.diff(prec, 1)
        # finding the biggest drop (+1 because diff reduces length, +1
        # because of indexing begins with 0)
        if len(discrete_difference) == 0:
            return 0

        # returning the coordinate of the biggest drop as the ideal
        # neighborhood size note that k indices the minority neighbors
        return np.argmin(discrete_difference) + 2

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
            return self.return_copies(X, y, "Sampling is not needed.")

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        K = min([len(X_min), self.K]) # pylint: disable=invalid-name

        # find K nearest neighbors of all samples
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=K,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nearestn.fit(X)
        ind = nearestn.kneighbors(X, return_distance=False)

        # indices of minority samples
        min_ind = np.where(y == self.min_label)[0]

        # Executing the algorithm
        # iteration through all minority samples
        # kappa stores the ideal neighborhood sizes for all minority samples
        kappa = []
        for _, min_idx in enumerate(min_ind):
            kappa.append(self.calculate_kappa(min_idx, ind[min_idx], X, y))

        # finding nearest minority neighbors of minority samples
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=max(kappa) + 1,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nearestn.fit(X_min)
        ind = nearestn.kneighbors(X_min, return_distance=False)

        if np.sum(kappa) == 0:
            return self.return_copies(X, y, "No minority samples in nearest neighbors")

        # turning the neighbor adjecency matrix into a weight matrix
        # based on the kappa scores

        weights = ind.copy()
        weights[:,:]= 1.0
        for idx in range(weights.shape[0]):
            if kappa[idx] == 0:
                weights[idx, :]= 0.0
            else:
                weights[idx, (kappa[idx]+1):]= 0.0

        samples = self.sample_simplex(X_min,
                                        indices=ind,
                                        n_to_sample=n_to_sample,
                                        simplex_weights=weights)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'K': self.K,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
