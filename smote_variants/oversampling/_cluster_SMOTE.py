import numpy as np

from sklearn.cluster import KMeans

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['cluster_SMOTE']

class cluster_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{cluster_SMOTE,
                            author={Cieslak, D. A. and Chawla, N. V. and
                                        Striegel, A.},
                            booktitle={2006 IEEE International Conference
                                        on Granular Computing},
                            title={Combating imbalance in network
                                        intrusion datasets},
                            year={2006},
                            volume={},
                            number={},
                            pages={732-737},
                            keywords={Intelligent networks;Intrusion detection;
                                        Telecommunication traffic;Data mining;
                                        Computer networks;Data security;
                                        Machine learning;Counting circuits;
                                        Computer security;Humans},
                            doi={10.1109/GRC.2006.1635905},
                            ISSN={},
                            month={May}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=3,
                 *,
                 nn_params={},
                 n_clusters=3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_clusters (int): number of clusters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_components", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_clusters = n_clusters
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
                                  'n_clusters': [3, 5, 7, 9]}
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

        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        n_clusters = min([len(X_min), self.n_clusters])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self._random_state_init)
        kmeans.fit(X_min)
        cluster_labels = kmeans.labels_
        unique_labels = np.unique(cluster_labels)

        # creating nearest neighbors objects for each cluster
        cluster_indices = [np.where(cluster_labels == c)[0]
                           for c in unique_labels]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        def nneighbors(idx):
            n_neighbors = min([self.n_neighbors, len(cluster_indices[idx])])
            nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
            return nn.fit(X_min[cluster_indices[idx]])

        cluster_nns = [nneighbors(idx) for idx in range(len(cluster_indices))]

        if max([len(c) for c in cluster_indices]) <= 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "All clusters contain 1 element")
            return X.copy(), y.copy()

        # generating the samples
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.randint(len(cluster_indices))
            if len(cluster_indices[cluster_idx]) <= 1:
                continue
            random_idx = self.random_state.randint(
                len(cluster_indices[cluster_idx]))
            sample_a = X_min[cluster_indices[cluster_idx]][random_idx]
            indices = cluster_nns[cluster_idx].kneighbors(sample_a.reshape(1, -1), return_distance=False)
            sample_b_idx = self.random_state.choice(
                cluster_indices[cluster_idx][indices[0][1:]])
            sample_b = X_min[sample_b_idx]
            samples.append(self.sample_between_points(sample_a, sample_b))

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
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

