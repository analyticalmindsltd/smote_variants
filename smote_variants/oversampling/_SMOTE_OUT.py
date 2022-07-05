import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_OUT']

class SMOTE_OUT(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                      title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An
                                enhancement strategy to handle imbalance in
                                data level},
                      author={Fajri Koto},
                      journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                      year={2014},
                      pages={280-284}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
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
        self.nn_params = nn_params
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        # nearest neighbors among minority points
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nn_min= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
        nn_min.fit(X_min)

        min_indices = nn_min.kneighbors(X_min, return_distance=False)
        # nearest neighbors among majority points
        n_neighbors = min([len(X_maj), self.n_neighbors+1])
        nn_maj= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
        nn_maj.fit(X_maj)
        maj_indices = nn_maj.kneighbors(X_min, return_distance=False)

        # generate samples
        samples = []
        for _ in range(n_to_sample):
            # implementation of Algorithm 1 in the paper
            random_idx = self.random_state.choice(
                np.arange(len(minority_indices)))
            u = X[minority_indices[random_idx]]
            v = X_maj[self.random_state.choice(maj_indices[random_idx])]
            dif1 = u - v
            uu = u + self.random_state.random_sample()*0.3*dif1
            x = X_min[self.random_state.choice(min_indices[random_idx][1:])]
            dif2 = uu - x
            w = x + self.random_state.random_sample()*0.5*dif2

            samples.append(w)

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
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
