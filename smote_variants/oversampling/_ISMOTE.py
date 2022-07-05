import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['ISMOTE']

class ISMOTE(OverSampling):
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

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 minority_weight=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            minority_weight (float): weight parameter according to the paper
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(minority_weight, "minority_weight", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.minority_weight = minority_weight
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        n_to_sample = int((len(X_maj) - len(X_min))/2 + 0.5)

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # computing distances of majority samples from minority ones
        nn = NearestNeighborsWithMetricTensor(n_neighbors=len(X_min), 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_maj)

        # sort majority instances in descending order by their mean distance
        # from minority samples
        to_sort = zip(np.arange(len(X_maj)), np.mean(dist, axis=1))
        ind_sorted, dist_sorted = zip(*sorted(to_sort, key=lambda x: -x[1]))

        # remove the ones being farthest from the minority samples
        X_maj = X_maj[list(ind_sorted[n_to_sample:])]

        # construct new dataset
        X_new = np.vstack([X_maj, X_min])
        y_new = np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min))])

        X_min = X_new[y_new == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_new), self.n_neighbors + 1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_new)
        dist, ind = nn.kneighbors(X_min)

        # do the oversampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)))
            y_idx = self.random_state.choice(ind[idx][1:])

            # different generation scheme depending on the class label
            if y_new[y_idx] == self.min_label:
                diff = (X_new[y_idx] - X_min[idx])
                r = self.random_state.random_sample()
                samples.append(X_min[idx] + r * diff * self.minority_weight)
            else:
                diff = (X_new[y_idx] - X_min[idx])
                r = self.random_state.random_sample()
                sample = X_min[idx] + r * diff * (1.0 - self.minority_weight)
                samples.append(sample)

        return (np.vstack([X_new, np.vstack(samples)]),
                np.hstack([y_new, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'minority_weight': self.minority_weight,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

