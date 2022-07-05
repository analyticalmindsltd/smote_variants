import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['kmeans_SMOTE']

class kmeans_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{kmeans_smote,
                        title = "Improving imbalanced learning through a
                                    heuristic oversampling method based
                                    on k-means and SMOTE",
                        journal = "Information Sciences",
                        volume = "465",
                        pages = "1 - 20",
                        year = "2018",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2018.06.056",
                        author = "Georgios Douzas and Fernando Bacao and
                                    Felix Last",
                        keywords = "Class-imbalanced learning, Oversampling,
                                    Classification, Clustering, Supervised
                                    learning, Within-class imbalance"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_clusters=10,
                 irt=2.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_clusters (int): number of clusters
            irt (float): imbalanced ratio threshold
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_clusters", 1)
        self.check_greater_or_equal(irt, "irt", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_clusters = n_clusters
        self.irt = irt
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
                                  'n_clusters': [2, 5, 10, 20, 50],
                                  'irt': [0.5, 0.8, 1.0, 1.5]}
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # applying kmeans clustering to all data
        n_clusters = min([self.n_clusters, len(X)])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self._random_state_init)
        kmeans.fit(X)

        # extracting clusters
        labels = kmeans.labels_
        clusters = [np.where(labels == li)[0] for li in range(n_clusters)]

        # cluster filtering
        def cluster_filter(c):
            numerator = np.sum(y[c] == self.maj_label) + 1
            denominator = np.sum(y[c] == self.min_label) + 1
            n_minority = np.sum(y[c] == self.min_label)
            return numerator/denominator < self.irt and n_minority > 1

        filt_clusters = [c for c in clusters if cluster_filter(c)]

        if len(filt_clusters) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "number of clusters after filtering is 0")
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # Step 2 in the paper
        sparsity = []
        nearest_neighbors = []
        cluster_minority_ind = []
        for c in filt_clusters:
            # extract minority indices in the cluster
            minority_ind = c[y[c] == self.min_label]
            cluster_minority_ind.append(minority_ind)
            # compute distance matrix of minority samples in the cluster
            dm = pairwise_distances_mahalanobis(X[minority_ind], tensor=nn_params['metric_tensor'])
            min_count = len(minority_ind)
            # compute the average of distances
            avg_min_dist = (np.sum(dm) - dm.trace()) / \
                (len(minority_ind)**2 - len(minority_ind))
            # compute sparsity (Step 4)
            sparsity.append(avg_min_dist**len(X[0])/min_count)
            # extract the nearest neighbors graph
            n_neighbors = min([len(minority_ind), self.n_neighbors + 1])
            nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
            nn.fit(X[minority_ind])
            nearest_neighbors.append(nn.kneighbors(X[minority_ind]))

        # Step 5 - compute density of sampling
        weights = sparsity/np.sum(sparsity)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # choose random cluster index and random minority element
            clust_ind = self.random_state.choice(
                np.arange(len(weights)), p=weights)
            idx = self.random_state.randint(
                len(cluster_minority_ind[clust_ind]))
            base_idx = cluster_minority_ind[clust_ind][idx]
            # choose random neighbor
            neighbor_cluster_indices = nearest_neighbors[clust_ind][1][idx][1:]
            domain = cluster_minority_ind[clust_ind][neighbor_cluster_indices]
            neighbor_idx = self.random_state.choice(domain)
            # sample
            X_a = X[base_idx]
            X_b = X[neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

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
                'n_clusters': self.n_clusters,
                'irt': self.irt,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

