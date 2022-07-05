import numpy as np

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                                            MetricTensor)
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['ADASYN']

class ADASYN(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{adasyn,
                          author={He, H. and Bai, Y. and Garcia,
                                    E. A. and Li, S.},
                          title={{ADASYN}: adaptive synthetic sampling
                                    approach for imbalanced learning},
                          booktitle={Proceedings of IJCNN},
                          year={2008},
                          pages={1322--1328}
                        }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_density_based,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 beta=1.0,
                 *,
                 d_th=0.9,
                 nn_params={},
                 n_jobs=1,
                 random_state=None,
                 proportion=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            d_th (float): tolerated deviation level from balancedness
            beta (float): target level of balancedness, same as proportion
                            in other techniques
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
            proportion (float): same as beta, for convenience
        """
        super().__init__()

        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_greater_or_equal(d_th, 'd_th', 0)
        self.check_greater_or_equal(beta, 'beta', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.d_th = d_th
        self.beta = proportion or beta
        self.proportion = self.beta
        self.nn_params = nn_params
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7, 9],
                                  'd_th': [0.9],
                                  'proportion': [2.0, 1.5, 1.0, 0.75, 0.5, 0.25]}
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

        # extracting minority samples
        X_min = X[y == self.min_label]

        # checking if sampling is needed
        m_min = len(X_min)
        m_maj = len(X) - m_min

        n_to_sample = (m_maj - m_min)*self.beta

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        d = float(m_min)/m_maj
        if d > self.d_th:
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model to all samples
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        indices = nn.kneighbors(X_min, return_distance=False)

        # determining the distribution of points to be generated
        r = []
        for i in range(len(indices)):
            r.append(sum(y[indices[i][1:]] ==
                         self.maj_label)/self.n_neighbors)
        r = np.array(r)
        if sum(r) > 0:
            r = r/sum(r)

        if any(np.isnan(r)) or sum(r) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "not enough non-noise samples for oversampling")
            return X.copy(), y.copy()

        # fitting nearest neighbors models to minority samples
        n_neigh = min([len(X_min), self.n_neighbors + 1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neigh, 
                                                        n_jobs=self.n_jobs, 
                                                        **nn_params)
        nn.fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)

        # sampling points
        base_indices = self.random_state.choice(
            list(range(len(X_min))), size=int(n_to_sample), p=r)
        neighbor_indices = self.random_state.choice(
            list(range(1, n_neigh)), int(n_to_sample))

        X_base = X_min[base_indices]
        X_neighbor = X_min[indices[base_indices, neighbor_indices]]
        diff = X_neighbor - X_base
        r = self.random_state.rand(int(n_to_sample), 1)

        samples = X_base + np.multiply(r, diff)

        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*int(n_to_sample))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'd_th': self.d_th,
                'beta': self.beta,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init,
                'proportion': self.proportion}
