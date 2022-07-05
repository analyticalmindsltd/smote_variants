import numpy as np

from sklearn.metrics import pairwise_distances

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['NRSBoundary_SMOTE']

class NRSBoundary_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{nrsboundary_smote,
                    author= {Feng, Hu and Hang, Li},
                    title= {A Novel Boundary Oversampling Algorithm Based on
                            Neighborhood Rough Set Model: NRSBoundary-SMOTE},
                    journal= {Mathematical Problems in Engineering},
                    year= {2013},
                    pages= {10},
                    doi= {10.1155/2013/694809},
                    url= {http://dx.doi.org/10.1155/694809}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 w=0.005,
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
            w (float): used to set neighborhood radius
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(w, "w", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.w = w
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
                                  'w': [0.005, 0.01, 0.05]}
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
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # step 1
        bound_set = []
        pos_set = []

        # step 2
        X_min_indices = np.where(y == self.min_label)[0]
        X_min = X[X_min_indices]

        # step 3
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        dm = pairwise_distances_mahalanobis(X, X, nn_params['metric_tensor'])
        d_max = np.max(dm, axis=1)
        max_dist = np.max(dm)
        np.fill_diagonal(dm, max_dist)
        d_min = np.min(dm, axis=1)

        delta = d_min + self.w*(d_max - d_min)

        # number of neighbors is not interesting here, as we use the
        # radius_neighbors function to extract the neighbors in a given radius
        n_neighbors = min([self.n_neighbors + 1, len(X)])

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        for i in range(len(X)):
            indices = nn.radius_neighbors(X[i].reshape(1, -1),
                                          delta[i],
                                          return_distance=False)

            n_minority = np.sum(y[indices[0]] == self.min_label)
            n_majority = np.sum(y[indices[0]] == self.maj_label)
            if y[i] == self.min_label and not n_minority == len(indices[0]):
                bound_set.append(i)
            elif y[i] == self.maj_label and n_majority == len(indices[0]):
                pos_set.append(i)

        bound_set = np.array(bound_set)
        pos_set = np.array(pos_set)

        if len(pos_set) == 0 or len(bound_set) == 0:
            return X.copy(), y.copy()

        # step 4 and 5
        # computing the nearest neighbors of the bound set from the
        # minority set
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        indices = nn.kneighbors(X[bound_set], return_distance=False)

        # do the sampling
        samples = []
        trials = 0
        w = self.w
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(len(bound_set))
            random_neighbor_idx = self.random_state.choice(indices[idx][1:])
            x_new = self.sample_between_points(
                X[bound_set[idx]], X_min[random_neighbor_idx])

            # checking the conflict
            dist_from_pos_set = np.linalg.norm(X[pos_set] - x_new, axis=1)
            if np.all(dist_from_pos_set > delta[pos_set]):
                # no conflict
                samples.append(x_new)
            trials = trials + 1
            if trials > 1000 and len(samples) == 0:
                trials = 0
                w = w*0.9

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
                'w': self.w,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

