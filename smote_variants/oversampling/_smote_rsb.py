"""
This module implements the SMOTE_RSB method.
"""

import numpy as np

from sklearn.metrics import pairwise_distances

from ..base import coalesce, coalesce_dict
from ..base import OverSampling
from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_RSB']

class SMOTE_RSB(OverSampling):
    """
    References:
        * BibTex::

            @Article{smote_rsb,
                    author="Ramentol, Enislay
                    and Caballero, Yail{\'e}
                    and Bello, Rafael
                    and Herrera, Francisco",
                    title="SMOTE-RSB*: a hybrid preprocessing approach
                            based on oversampling and undersampling for
                            high imbalanced data-sets using SMOTE and
                            rough sets theory",
                    journal="Knowledge and Information Systems",
                    year="2012",
                    month="Nov",
                    day="01",
                    volume="33",
                    number="2",
                    pages="245--265",
                    issn="0219-3116",
                    doi="10.1007/s10115-011-0465-6",
                    url="https://doi.org/10.1007/s10115-011-0465-6"
                    }

    Notes:
        * I think the description of the algorithm in Fig 5 of the paper
            is not correct. The set "resultSet" is initialized with the
            original instances, and then the While loop in the Algorithm
            run until resultSet is empty, which never holds. Also, the
            resultSet is only extended in the loop. Our implementation
            is changed in the following way: we generate twice as many
            instances are required to balance the dataset, and repeat
            the loop until the number of new samples added to the training
            set is enough to balance the dataset.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

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
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
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

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
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
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def normalization_factor(self, X_samp):
        """
        Construct the normalization factors.

        Args:
            X_samp (np.array): the generated samples

        Returns:
            np.array: the normalization factors
        """
        # Step 3: first the data is normalized
        maximums = np.max(X_samp, axis=0)
        minimums = np.min(X_samp, axis=0)

        # normalize X_new and X_maj
        norm_factor = maximums - minimums
        null_mask = norm_factor == 0
        n_null = np.sum(null_mask)
        fixed = np.max(np.vstack([maximums[null_mask], np.repeat(1, n_null)]),
                       axis=0)

        norm_factor[null_mask] = fixed

        return norm_factor

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        X_maj = X[y == self.maj_label]
        X_min = X[y == self.min_label]

        # Step 1: do the sampling
        smote = SMOTE(proportion=self.proportion,
                      n_neighbors=self.n_neighbors,
                      nn_params=self.nn_params,
                      ss_params=self.ss_params,
                      n_jobs=self.n_jobs,
                      random_state=self._random_state_init)

        X_samp, _ = smote.sample(X, y)
        X_samp = X_samp[len(X):]

        if len(X_samp) == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # Step 2: (original will be added later)
        result_set = np.zeros(shape=(0, X.shape[1]))

        norm_factor = self.normalization_factor(X_samp)

        # compute similarity matrix
        similarity_matrix = 1.0 - pairwise_distances(X_samp/norm_factor,
                                                     X_maj/norm_factor,
                                                     metric='minkowski',
                                                     p=1) / X.shape[1]

        # Step 4: counting the similar examples
        similarity_value = 0.4

        already_added = np.repeat(False, len(X_samp))
        while (result_set.shape[0] < X_maj.shape[0] - X_min.shape[0] \
                                            and similarity_value <= 0.9):
            conts = np.sum(similarity_matrix > similarity_value, axis=1)
            mask = (conts == 0) & (~ already_added)
            result_set = np.vstack([result_set, X_samp[mask]])
            already_added[mask] = True

            similarity_value = similarity_value + 0.05

        result_set = result_set[:np.min([result_set.shape[0],
                                        X_maj.shape[0] - X_min.shape[0]])]

        # Step 5: returning the results depending the number of instances
        # added to the result set
        return (np.vstack([X, result_set]),
                np.hstack([y, np.repeat(self.min_label,
                                        len(result_set))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
