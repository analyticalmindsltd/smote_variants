import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['Lee']

class Lee(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{lee,
                             author = {Lee, Jaedong and Kim,
                                 Noo-ri and Lee, Jee-Hyong},
                             title = {An Over-sampling Technique with Rejection
                                        for Imbalanced Class Learning},
                             booktitle = {Proceedings of the 9th International
                                            Conference on Ubiquitous
                                            Information Management and
                                            Communication},
                             series = {IMCOM '15},
                             year = {2015},
                             isbn = {978-1-4503-3377-1},
                             location = {Bali, Indonesia},
                             pages = {102:1--102:6},
                             articleno = {102},
                             numpages = {6},
                             doi = {10.1145/2701126.2701181},
                             acmid = {2701181},
                             publisher = {ACM},
                             address = {New York, NY, USA},
                             keywords = {data distribution, data preprocessing,
                                            imbalanced problem, rejection rule,
                                            synthetic minority oversampling
                                            technique}
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 rejection_level=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbor
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            rejection_level (float): the rejection level of generated samples,
                                        if the fraction of majority labels in
                                        the local environment is higher than
                                        this number, the generated point is
                                        rejected
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(rejection_level, "rejection_level", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.rejection_level = rejection_level
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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
                                  'rejection_level': [0.3, 0.5, 0.7]}
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

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors models to find neighbors of minority
        # samples in the total data and in the minority datasets
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        dist, ind = nn.kneighbors(X_min)

        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn_min= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
        nn_min.fit(X_min)
        dist_min, ind_min = nn_min.kneighbors(X_min)

        # do the sampling, we implemented a continuous tweaking of rejection
        # levels in order to fix situations when no unrejectable data can
        # be can be generated
        samples = []
        passed = 0
        trial = 0
        rejection_level = self.rejection_level
        while len(samples) < n_to_sample:
            # checking if we managed to generate a single data in 1000 trials
            if passed == trial and passed > 1000:
                rejection_level = rejection_level + 0.1
                trial = 0
                passed = 0
            trial = trial + 1
            # generating random point
            idx = self.random_state.randint(len(X_min))
            random_neighbor_idx = self.random_state.choice(ind_min[idx][1:])
            X_a = X_min[idx]
            X_b = X_min[random_neighbor_idx]
            random_point = self.sample_between_points(X_a, X_b)
            # checking if the local environment is above the rejection level
            dist_new, ind_new = nn.kneighbors(random_point.reshape(1, -1))
            maj_frac = np.sum(y[ind_new][:-1] == self.maj_label)/self.n_neighbors
            if maj_frac < rejection_level:
                samples.append(random_point)
            else:
                passed = passed + 1

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'rejection_level': self.rejection_level,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
