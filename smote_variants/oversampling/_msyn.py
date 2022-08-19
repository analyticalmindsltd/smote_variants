"""
This module implements the MSYN method.
"""

import numpy as np

from ..base import (NearestNeighborsWithMetricTensor,
                            pairwise_distances_mahalanobis, coalesce,
                            coalesce_dict)
from ..base import OverSampling
from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['MSYN']

class MSYN(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{msyn,
                            author="Fan, Xiannian
                            and Tang, Ke
                            and Weise, Thomas",
                            editor="Huang, Joshua Zhexue
                            and Cao, Longbing
                            and Srivastava, Jaideep",
                            title="Margin-Based Over-Sampling Method for
                                    Learning from Imbalanced Datasets",
                            booktitle="Advances in Knowledge Discovery and
                                        Data Mining",
                            year="2011",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="309--320",
                            abstract="Learning from imbalanced datasets has
                                        drawn more and more attentions from
                                        both theoretical and practical aspects.
                                        Over- sampling is a popular and simple
                                        method for imbalanced learning. In this
                                        paper, we show that there is an
                                        inherently potential risk associated
                                        with the over-sampling algorithms in
                                        terms of the large margin principle.
                                        Then we propose a new synthetic over
                                        sampling method, named Margin-guided
                                        Synthetic Over-sampling (MSYN), to
                                        reduce this risk. The MSYN improves
                                        learning with respect to the data
                                        distributions guided by the
                                        margin-based rule. Empirical study
                                        verities the efficacy of MSYN.",
                            isbn="978-3-642-20847-8"
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 pressure=1.5,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 proportion=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            pressure (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in the SMOTE sampling
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
        self.check_greater_or_equal(pressure, 'pressure', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.pressure = proportion or pressure
        self.proportion = self.pressure
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
        parameter_combinations = {'proportion': [2.5, 2.0, 1.5, 1.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_Delta_P_N(self, *, # pylint: disable=invalid-name
                            nearest_hit_dist,
                            nearest_miss_dist,
                            min_indices,
                            maj_indices,
                            distances,
                            theta_min,
                            theta_maj):
        """
        Determine the Delta_P and Delta_N quantities.

        Args:
            nearest_hit_dist (np.array): the nearest hit distances
            nearest_miss_dist (np.array): the nearest miss distances
            min_indices (np.array): the minority indices
            maj_indices (np.array): the majority indices
            distances (np.array): the distance matrix
            theta_min (np.array): the theta_min array
            theta_maj (np.array): the theta_maj array

        Returns:
            np.array, np.array: the Delta_P and Delta_N arrays
        """

        mask = (nearest_hit_dist[min_indices] < distances[min_indices].T).T
        nearest_hit_dist_min = np.where(np.logical_not(mask).T, distances[min_indices].T,
                                        nearest_hit_dist[min_indices]).T
        nearest_miss_dist_min = nearest_miss_dist[min_indices]
        nearest_hit_dist_maj = nearest_hit_dist[maj_indices]
        mask = (nearest_miss_dist[maj_indices] < distances[maj_indices].T).T
        nearest_miss_dist_maj = np.where(np.logical_not(mask).T, distances[maj_indices].T,
                                            nearest_miss_dist[maj_indices]).T

        theta_x_min = 0.5*(nearest_miss_dist_min - nearest_hit_dist_min.T)
        theta_x_maj = 0.5*(nearest_miss_dist_maj.T - nearest_hit_dist_maj)

        return (np.sum(theta_x_min - theta_min, axis=1),
                np.sum(theta_x_maj - theta_maj, axis=1))

    def determine_nearest_hit_miss_dist(self, X, y, nn_params):
        """
        Determine the nearest hit/miss distances

        Args:
            X (np.array): the feature vectors
            y (np.array): the target labels
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array, np.array: the nearest hit/miss distances
        """
        # Compute nearest hit and miss for both classes
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=len(X),
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        dist, ind = nnmt.kneighbors(X)

        # computing nearest hit and miss distances, these will be used to
        # compute thetas
        hit_mask = y[ind[:, 1:]] == y[ind[:, 0]][:, None]
        nearest_hit_dist = dist[:, 1:][np.arange(y.shape[0]),
                                        np.argmax(hit_mask, axis=1)]
        nearest_miss_dist = dist[:, 1:][np.arange(y.shape[0]),
                                        np.argmax(np.logical_not(hit_mask),
                                                    axis=1)]

        return nearest_hit_dist, nearest_miss_dist

    def determine_f_3(self, *, X, y, X_new, nn_params):
        """
        Determine the f_3 scores.

        Args:
            X (np.array): all training vectors
            y (np.array): all labels
            X_new (np.array): the new samples
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array: the f_3 scores
        """
        min_indices = np.where(y == self.min_label)[0]
        maj_indices = np.where(y == self.maj_label)[0]

        nearest_hit_dist, nearest_miss_dist = \
            self.determine_nearest_hit_miss_dist(X, y, nn_params)

        # computing the thetas without new samples being involved
        theta_A_sub_alpha = 0.5*(nearest_miss_dist - nearest_hit_dist) # pylint: disable=invalid-name
        theta_min = theta_A_sub_alpha[min_indices]
        theta_maj = theta_A_sub_alpha[maj_indices]

        distances = pairwise_distances_mahalanobis(X,
                                                    Y=X_new,
                                                    tensor=nn_params['metric_tensor'])

        Delta_P, Delta_N = self.determine_Delta_P_N(nearest_hit_dist=nearest_hit_dist,  # pylint: disable=invalid-name
                                                    nearest_miss_dist=nearest_miss_dist,
                                                    min_indices=min_indices,
                                                    maj_indices=maj_indices,
                                                    distances=distances,
                                                    theta_min=theta_min,
                                                    theta_maj=theta_maj)

        return -Delta_N/(Delta_P + 0.01)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.pressure)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed.")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # generating samples
        smote = SMOTE(proportion=self.pressure*3,
                      n_neighbors=self.n_neighbors,
                      nn_params=nn_params,
                      ss_params=self.ss_params,
                      n_jobs=self.n_jobs,
                      random_state=self._random_state_init)

        X_res, y_res = smote.sample(X, y) # pylint: disable=invalid-name
        X_new, _ = X_res[len(X):], y_res[len(X):]

        f_3 = self.determine_f_3(X=X, y=y,
                                X_new=X_new,
                                nn_params=nn_params)

        # determining the elements with the minimum f_3 scores to add
        _, new_ind = zip(*sorted(zip(f_3, np.arange(len(f_3))), key=lambda x: x[0]))
        new_ind = list(new_ind[:n_to_sample])

        return (np.vstack([X, X_new[new_ind]]),
                np.hstack([y, np.repeat(self.min_label, len(new_ind))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'pressure': self.pressure,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self),
                'proportion': self.proportion}
