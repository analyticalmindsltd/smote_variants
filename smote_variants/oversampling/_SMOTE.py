import numpy as np

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                                            MetricTensor, generate_samples,
                                                            AdditionalItems)
from sklearn.neighbors import NearestNeighbors

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE']

class SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{smote,
                author={Chawla, N. V. and Bowyer, K. W. and Hall, L. O. and
                            Kegelmeyer, W. P.},
                title={{SMOTE}: synthetic minority over-sampling technique},
                journal={Journal of Artificial Intelligence Research},
                volume={16},
                year={2002},
                pages={321--357}
              }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the SMOTE object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0
            means that after sampling the number of minority samples will
                                 be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                technique
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

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params= nn_params
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
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

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ + ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the model
        n_neighbors = min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_min)
        ind= nn.kneighbors(X_min, return_distance=False)

        if n_to_sample == 0:
            return X.copy(), y.copy()

        # generating samples
        base_indices = self.random_state.choice(list(range(len(X_min))),
                                                n_to_sample)
        neighbor_indices = self.random_state.choice(list(range(1, n_neighbors)),
                                                    n_to_sample)

        X_base = X_min[base_indices]
        X_neighbors = X_min[ind[base_indices, neighbor_indices]]
        random_offsets= self.random_state.rand(n_to_sample)

        samples= X_base + random_offsets[:,None]*(X_neighbors - X_base)
        
        return (np.vstack([X, samples]),
                np.hstack([y, np.hstack([self.min_label]*n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
