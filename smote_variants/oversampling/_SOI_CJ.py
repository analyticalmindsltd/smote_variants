import numpy as np

from sklearn.metrics import pairwise_distances

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ..noise_removal import EditedNearestNeighbors
from .._base import mode
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SOI_CJ']

class SOI_CJ(OverSampling):
    """
    References:
        * BibTex::

            @article{soi_cj,
                    author = {Sánchez, Atlántida I. and Morales, Eduardo and
                                Gonzalez, Jesus},
                    year = {2013},
                    month = {01},
                    pages = {},
                    title = {Synthetic Oversampling of Instances Using
                                Clustering},
                    volume = {22},
                    booktitle = {International Journal of Artificial
                                    Intelligence Tools}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 method='interpolation',
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of nearest neighbors in the SMOTE
                                sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            method (str): 'interpolation'/'jittering'
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_isin(method, 'method', ['interpolation', 'jittering'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.method = method
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
                                  'method': ['interpolation', 'jittering']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def clustering(self, X, y):
        """
        Implementation of the clustering technique described in the paper.

        Args:
            X (np.matrix): array of training instances
            y (np.array): target labels

        Returns:
            list(set): list of minority clusters
        """
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        nn_all= NearestNeighborsWithMetricTensor(n_jobs=self.n_jobs, 
                                                    **nn_params)
        nn_all.fit(X)

        X_min = X[y == self.min_label]

        # extract nearest neighbors of all samples from the set of
        # minority samples
        nn= NearestNeighborsWithMetricTensor(n_neighbors=len(X_min), 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        indices = nn.kneighbors(X_min, return_distance=False)

        # initialize clusters by minority samples
        clusters = []
        for i in range(len(X_min)):
            # empty cluster added
            clusters.append(set())
            # while the closest instance is from the minority class, adding it
            # to the cluster
            for j in indices[i]:
                if y[j] == self.min_label:
                    clusters[i].add(j)
                else:
                    break

        # cluster merging phase
        is_intersection = True
        while is_intersection:
            is_intersection = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # computing intersection
                    intersection = clusters[i].intersection(clusters[j])
                    if len(intersection) > 0:
                        is_intersection = True
                        # computing distance matrix
                        dm = pairwise_distances(
                            X[list(clusters[i])], X[list(clusters[j])])
                        # largest distance
                        max_dist_pair = np.where(dm == np.max(dm))
                        # elements with the largest distance
                        max_i = X[list(clusters[i])[max_dist_pair[0][0]]]
                        max_j = X[list(clusters[j])[max_dist_pair[1][0]]]

                        # finding midpoint and radius
                        mid_point = (max_i + max_j)/2.0
                        radius = np.linalg.norm(mid_point - max_i)

                        # extracting points within the hypersphare of
                        # radius "radius"
                        mid_point_reshaped = mid_point.reshape(1, -1)
                        ind = nn_all.radius_neighbors(mid_point_reshaped,
                                                      radius,
                                                      return_distance=False)

                        n_min = np.sum(y[ind[0]] == self.min_label)
                        if n_min > len(ind[0])/2:
                            # if most of the covered elements come from the
                            # minority class, merge clusters
                            clusters[i].update(clusters[j])
                            clusters[j] = set()
                        else:
                            # otherwise move the difference to the
                            # bigger cluster
                            if len(clusters[i]) > len(clusters[j]):
                                clusters[j].difference_update(intersection)
                            else:
                                clusters[i].difference_update(intersection)

        # returning non-empty clusters
        return [c for c in clusters if len(c) > 0]

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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        std_min = np.std(X_min, axis=0)

        # do the clustering
        _logger.info(self.__class__.__name__ + ": " + "Executing clustering")
        clusters = self.clustering(X, y)

        # filtering the clusters, at least two points in a cluster are needed
        # for both interpolation and jittering (due to the standard deviation)
        clusters_filtered = [list(c) for c in clusters if len(c) > 2]

        if len(clusters_filtered) > 0:
            # if there are clusters having at least 2 elements, do the sampling
            cluster_nums = [len(c) for c in clusters_filtered]
            cluster_weights = cluster_nums/np.sum(cluster_nums)
            cluster_stds = [np.std(X[clusters_filtered[i]], axis=0)
                            for i in range(len(clusters_filtered))]

            _logger.info(self.__class__.__name__ + ": " +
                         "Executing sample generation")
            samples = []
            while len(samples) < n_to_sample:
                cluster_idx = self.random_state.choice(
                    np.arange(len(clusters_filtered)), p=cluster_weights)
                if self.method == 'interpolation':
                    clust = clusters_filtered[cluster_idx]
                    idx_0, idx_1 = self.random_state.choice(clust,
                                                            2,
                                                            replace=False)
                    X_0, X_1 = X[idx_0], X[idx_1]
                    samples.append(
                        self.sample_between_points_componentwise(X_0, X_1))
                elif self.method == 'jittering':
                    clust_std = cluster_stds[cluster_idx]
                    std = np.min(np.vstack([std_min, clust_std]), axis=0)
                    clust = clusters_filtered[cluster_idx]
                    idx = self.random_state.choice(clust)
                    X_samp = self.sample_by_jittering_componentwise(X[idx],
                                                                    std)
                    samples.append(X_samp)

            return (np.vstack([X, samples]),
                    np.hstack([y, np.array([self.min_label]*len(samples))]))
        else:
            # otherwise fall back to standard smote
            _logger.warning(self.__class__.__name__ + ": " +
                            "No clusters with more than 2 elements")
            return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'method': self.method,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
