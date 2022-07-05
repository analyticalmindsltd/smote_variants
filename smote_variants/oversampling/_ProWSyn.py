import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['ProWSyn']

class ProWSyn(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{prowsyn,
                        author="Barua, Sukarna
                        and Islam, Md. Monirul
                        and Murase, Kazuyuki",
                        editor="Pei, Jian
                        and Tseng, Vincent S.
                        and Cao, Longbing
                        and Motoda, Hiroshi
                        and Xu, Guandong",
                        title="ProWSyn: Proximity Weighted Synthetic
                                        Oversampling Technique for
                                        Imbalanced Data Set Learning",
                        booktitle="Advances in Knowledge Discovery
                                    and Data Mining",
                        year="2013",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="317--328",
                        isbn="978-3-642-37456-2"
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
                 L=5,
                 theta=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            L (int): number of levels
            theta (float): smoothing factor in weight formula
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(L, "L", 1)
        self.check_greater_or_equal(theta, "theta", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.L = L
        self.theta = theta
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
                                  'L': [3, 5, 7],
                                  'theta': [0.1, 1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and
                                    target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        # Step 1 - a bit generalized
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            m = "Sampling is not needed"
            _logger.warning(self.__class__.__name__ + ": " + m)
            return X.copy(), y.copy()

        # Step 2
        P = np.where(y == self.min_label)[0]
        X_maj = X[y == self.maj_label]

        Ps = []
        proximity_levels = []

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # Step 3
        for i in range(self.L):
            if len(P) == 0:
                break
            # Step 3 a
            n_neighbors = min([len(P), self.n_neighbors])

            nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
            nn.fit(X[P])
            indices = nn.kneighbors(X_maj, return_distance=False)

            # Step 3 b
            P_i = np.unique(np.hstack([i for i in indices]))

            # Step 3 c - proximity levels are encoded in the Ps list index
            Ps.append(P[P_i])
            proximity_levels.append(i+1)

            # Step 3 d
            P = np.delete(P, P_i)

        # Step 4
        if len(P) > 0:
            Ps.append(P)

        # Step 5
        if len(P) > 0:
            proximity_levels.append(i)
            proximity_levels = np.array(proximity_levels)

        # Step 6
        weights = np.array([np.exp(-self.theta*(proximity_levels[i] - 1))
                            for i in range(len(proximity_levels))])
        # weights is the probability distribution of sampling in the
        # clusters identified
        weights = weights/np.sum(weights)

        suitable = False
        for i in range(len(weights)):
            if weights[i] > 0 and len(Ps[i]) > 1:
                suitable = True

        if not suitable:
            return X.copy(), y.copy()

        # do the sampling, from each cluster proportionally to the distribution
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(
                np.arange(len(weights)), p=weights)
            if len(Ps[cluster_idx]) > 1:
                random_idx1, random_idx2 = self.random_state.choice(
                    Ps[cluster_idx], 2, replace=False)
                samples.append(self.sample_between_points(
                    X[random_idx1], X[random_idx2]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'L': self.L,
                'theta': self.theta,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

