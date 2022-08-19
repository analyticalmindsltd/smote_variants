"""
This module implements the Stefanowski method.
"""

import numpy as np

from scipy.stats import mode

from ..base import coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['Stefanowski']

class Stefanowski(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{stefanowski,
                 author = {Stefanowski, Jerzy and Wilk, Szymon},
                 title = {Selective Pre-processing of Imbalanced Data for
                            Improving Classification Performance},
                 booktitle = {Proceedings of the 10th International Conference
                                on Data Warehousing and Knowledge Discovery},
                 series = {DaWaK '08},
                 year = {2008},
                 isbn = {978-3-540-85835-5},
                 location = {Turin, Italy},
                 pages = {283--292},
                 numpages = {10},
                 url = {http://dx.doi.org/10.1007/978-3-540-85836-2_27},
                 doi = {10.1007/978-3-540-85836-2_27},
                 acmid = {1430591},
                 publisher = {Springer-Verlag},
                 address = {Berlin, Heidelberg},
                }
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_sample_copy,
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 *,
                 strategy='weak_amp',
                 nn_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            strategy (str): 'weak_amp'/'weak_amp_relabel'/'strong_amp'
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state, checks={'min_n_min': 6})

        self.check_isin(strategy,
                        'strategy',
                        ['weak_amp', 'weak_amp_relabel', 'strong_amp'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        if not raw:
            return [{'strategy': 'weak_amp'},
                    {'strategy': 'weak_amp_relabel'},
                    {'strategy': 'strong_amp'}, ]

        return {'strategy': ['weak_amp', 'weak_amp_relabel', 'strong_amp']}

    def create_multiples(self, X, indices, numbers):
        """
        Create multiples from vectors.

        Args:
            X (np.array): base vectors
            indices (np.array): the indices of vectors to multiply
            numbers (np.array): the counts of multiplication

        Returns:
            np.array: the multipled vectors
        """
        samples = np.zeros(shape=(0, X.shape[1]))
        for idx, min_idx in enumerate(indices):
            samples = np.vstack([samples, np.tile(X[min_idx], (numbers[idx], 1))])
        return samples

    def weak_amp(self, *, X, y, safe_flag, indices):
        """
        Weak amplification.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            safe_flag (np.array): True if the vector is safe, False if noise
            indices (np.array): the neighborhood structure

        Returns:
            np.array: the generated samples
        """
        minority_indices = np.where(y == self.min_label)[0]

        samples = np.zeros((0, X.shape[1]))
        # weak mplification - the number of copies is the number of
        # majority nearest neighbors
        mask = ~ safe_flag[minority_indices]
        minority_masked = minority_indices[mask]
        k = np.sum((y[indices[minority_masked][:, 1:]] == self.maj_label) # pylint: disable=invalid-name
                        & (safe_flag[indices[minority_masked][:, 1:]]), axis=1)

        samples = np.vstack([samples,
                             self.create_multiples(X, minority_masked, k)])

        return samples

    def weak_amp_relabel(self, *, y, safe_flag, D, indices): # pylint: disable=invalid-name
        """
        Weak amplification relabeling.

        Args:
            y (np.array): all target labels
            safe_flag (np.array): True if the vector is safe, False if noise
            D (np.array): the D array
            indices (np.array): the neighborhood structure

        Returns:
            np.array, np.array: the relabeled y and the updated D array
        """
        minority_indices = np.where(y == self.min_label)[0]

        # relabling - noisy majority neighbors are relabelled to minority
        mask = ~ safe_flag[minority_indices]
        minority_masked = minority_indices[mask]
        ind_tmp = indices[minority_masked][:, 1:]
        ind_tmp = ind_tmp[(y[ind_tmp] == self.maj_label) & (~ safe_flag[ind_tmp])]
        ind = list(set(ind_tmp.flatten().tolist()))
        y[ind] = self.min_label
        D[ind] = False

        return y, D

    def strong_amp(self, *, X, y, safe_flag, indices, indices5):
        """
        Strong amplification.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            safe_flag (np.array): True if the vector is safe, False if noise
            indices (np.array): the neighborhood structure
            indices5 (np.array): the neighborhood structure with 5 neighbors

        Returns:
            np.array: the generated samples
        """
        minority_indices = np.where(y == self.min_label)[0]

        samples = np.zeros((0, X.shape[1]))
        # safe minority samples are copied as many times as many safe
        # majority samples are among the nearest neighbors
        mask = safe_flag[minority_indices]
        minority_masked = minority_indices[mask]
        k = np.sum((y[indices[minority_masked][:, 1:]] == self.maj_label) # pylint: disable=invalid-name
                        & (safe_flag[indices[minority_masked][:, 1:]]), axis=1)
        samples = np.vstack([samples,
                             self.create_multiples(X, minority_masked, k)])

        mask = ~ safe_flag[minority_indices]
        minority_masked = minority_indices[mask]
        correct_mask = mode(y[indices5[minority_masked, 1:]]).mode[:, 0] \
                                                        == y[minority_masked]

        k = np.sum((y[indices[minority_masked[correct_mask], 1:]] == self.maj_label) \
                    & safe_flag[indices[minority_masked[correct_mask], 1:]], axis=1) # pylint: disable=invalid-name
        samples = np.vstack([samples,
                             self.create_multiples(X, minority_masked[correct_mask], k)])

        k = np.sum((y[indices5[minority_masked[~correct_mask], 1:]] == self.maj_label) \
                    & safe_flag[indices5[minority_masked[~correct_mask], 1:]], axis=1) # pylint: disable=invalid-name
        samples = np.vstack([samples,
                             self.create_multiples(X, minority_masked[~correct_mask], k)])

        return samples

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        X_orig, y_orig = X, y

        # copying y as its values will change
        y = y.copy()
        # fitting the nearest neighbors model for noise filtering, 4 neighbors
        # instead of 3 as the closest neighbor to a point is itself
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=np.min([4, len(X)]),
                                               n_jobs=self.n_jobs,
                                               **nn_params)
        nnmt.fit(X)
        indices = nnmt.kneighbors(X, return_distance=False)

        # fitting the nearest neighbors model for sample generation,
        # 6 neighbors instead of 5 for the same reason
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=np.min([6, len(X)]),
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        indices5 = nnmt.kneighbors(X, return_distance=False)

        # determining noisy and safe flags
        safe_flag = mode(y[indices[:, 1:]], axis=1).mode[:, 0] == y
        D = (y == self.maj_label) & (~ safe_flag) # pylint: disable=invalid-name

        samples = np.zeros((0, X.shape[1]))

        if self.strategy in ['weak_amp', 'weak_amp_relabel']:
            samples = np.vstack([samples,
                                 self.weak_amp(X=X,
                                                y=y,
                                                safe_flag=safe_flag,
                                                indices=indices)])

        if self.strategy == 'weak_amp_relabel':
            y, D = self.weak_amp_relabel(y=y, # pylint: disable=invalid-name
                                        safe_flag=safe_flag,
                                        D=D,
                                        indices=indices)

        if self.strategy == 'strong_amp':
            samples = np.vstack([samples,
                                 self.strong_amp(X=X,
                                                y=y,
                                                safe_flag=safe_flag,
                                                indices=indices,
                                                indices5=indices5)])

        to_remove = np.where(D)[0]

        X_noise_removed = np.delete(X, to_remove, axis=0) # pylint: disable=invalid-name
        y_noise_removed = np.delete(y, to_remove, axis=0)

        if np.unique(y_noise_removed).shape[0] < 2:
            return self.return_copies(X_orig, y_orig, "one class has been completely removed")

        return (np.vstack([X_noise_removed, samples]),
                np.hstack([y_noise_removed,
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'strategy': self.strategy,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
