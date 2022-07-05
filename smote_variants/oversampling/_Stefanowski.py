import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._base import mode
from ._SMOTE import SMOTE

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
                 nn_params={}, 
                 n_jobs=1, 
                 random_state=None):
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
        super().__init__()

        self.check_isin(strategy,
                        'strategy',
                        ['weak_amp', 'weak_amp_relabel', 'strong_amp'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.nn_params = nn_params
        self.n_jobs = n_jobs

        # this method does not maintain randomness, the parameter is
        # introduced for the compatibility of interfaces
        self.set_random_state(random_state)

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
        else:
            return {'strategy': ['weak_amp', 'weak_amp_relabel', 'strong_amp']}

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if self.class_stats[self.min_label] < 6:
            m = ("The number of minority samples (%d) is not"
                 " enough for sampling")
            m = m % (self.class_stats[self.min_label])
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # copying y as its values will change
        y = y.copy()
        # fitting the nearest neighbors model for noise filtering, 4 neighbors
        # instead of 3 as the closest neighbor to a point is itself
        nn= NearestNeighborsWithMetricTensor(n_neighbors=min(4, len(X)), 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)

        # fitting the nearest neighbors model for sample generation,
        # 6 neighbors instead of 5 for the same reason
        nn5= NearestNeighborsWithMetricTensor(n_neighbors=min(6, len(X)), 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn5.fit(X)
        indices5 = nn5.kneighbors(X, return_distance=False)

        # determining noisy and safe flags
        flags = []
        for i in range(len(indices)):
            if mode(y[indices[i][1:]]) == y[i]:
                flags.append('safe')
            else:
                flags.append('noisy')
        flags = np.array(flags)

        D = (y == self.maj_label) & (flags == 'noisy')
        minority_indices = np.where(y == self.min_label)[0]

        samples = []
        if self.strategy == 'weak_amp' or self.strategy == 'weak_amp_relabel':
            # weak mplification - the number of copies is the number of
            # majority nearest neighbors
            for i in minority_indices:
                if flags[i] == 'noisy':
                    k = np.sum(np.logical_and(
                        y[indices[i][1:]] == self.maj_label,
                        flags[indices[i][1:]] == 'safe'))
                    for _ in range(k):
                        samples.append(X[i])
        if self.strategy == 'weak_amp_relabel':
            # relabling - noisy majority neighbors are relabelled to minority
            for i in minority_indices:
                if flags[i] == 'noisy':
                    for j in indices[i][1:]:
                        if y[j] == self.maj_label and flags[j] == 'noisy':
                            y[j] = self.min_label
                            D[j] = False
        if self.strategy == 'strong_amp':
            # safe minority samples are copied as many times as many safe
            # majority samples are among the nearest neighbors
            for i in minority_indices:
                if flags[i] == 'safe':
                    k = np.sum(np.logical_and(
                        y[indices[i][1:]] == self.maj_label,
                        flags[indices[i][1:]] == 'safe'))
                    for _ in range(k):
                        samples.append(X[i])
            # if classified correctly by knn(5), noisy minority samples are
            # amplified by creating as many copies as many save majority
            # samples in its neighborhood are present otherwise amplify
            # based on the 5 neighborhood
            for i in minority_indices:
                if flags[i] == 'noisy':
                    if mode(y[indices5[i][1:]]) == y[i]:
                        k = np.sum(np.logical_and(
                            y[indices[i][1:]] == self.maj_label,
                            flags[indices[i][1:]] == 'safe'))
                    else:
                        k = np.sum(np.logical_and(
                            y[indices5[i][1:]] == self.maj_label,
                            flags[indices5[i][1:]] == 'safe'))
                    for _ in range(k):
                        samples.append(X[i])

        to_remove = np.where(D)[0]

        X_noise_removed = np.delete(X, to_remove, axis=0)
        y_noise_removed = np.delete(y, to_remove, axis=0)

        if len(samples) == 0 and len(X_noise_removed) > 10:
            m = "no samples to add"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X_noise_removed, y_noise_removed
        elif len(samples) == 0:
            m = "all samples removed as noise, returning the original dataset"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        return (np.vstack([X_noise_removed,
                           np.vstack(samples)]),
                np.hstack([y_noise_removed,
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'strategy': self.strategy,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs}
