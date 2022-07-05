import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['DBSMOTE']

class DBSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{dbsmote,
                        author="Bunkhumpornpat, Chumphol
                        and Sinapiromsaran, Krung
                        and Lursinsap, Chidchanok",
                        title="DBSMOTE: Density-Based Synthetic Minority
                                Over-sampling TEchnique",
                        journal="Applied Intelligence",
                        year="2012",
                        month="Apr",
                        day="01",
                        volume="36",
                        number="3",
                        pages="664--684",
                        issn="1573-7497",
                        doi="10.1007/s10489-011-0287-y",
                        url="https://doi.org/10.1007/s10489-011-0287-y"
                        }

    Notes:
        * Standardization is needed to use absolute eps values.
        * The clustering is likely to identify all instances as noise, fixed
            by recursive call with increaseing eps.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 eps=0.8,
                 min_samples=3,
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
            eps (float): eps parameter of DBSCAN
            min_samples (int): min_samples parameter of DBSCAN
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
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(min_samples, "min_samples", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.eps = eps
        self.min_samples = min_samples
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
                                  'eps': [0.5, 0.8, 1.2],
                                  'min_samples': [1, 3, 5]}
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

        ss = StandardScaler().fit(X)
        X_ss = ss.transform(X)

        # doing the clustering using DBSCAN
        X_min = X_ss[y == self.min_label]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=self.n_jobs).fit(X_min)
        labels = db.labels_
        num_labels = np.max(labels)+1

        if num_labels == 0:
            # adjusting the parameters if no clusters were identified
            message = ("Number of clusters is 0, trying to increase eps and "
                       "decrease min_samples")
            _logger.info(self.__class__.__name__ + ": " + message)
            if self.eps >= 2 or self.min_samples <= 2:
                message = ("Number of clusters is 0, can't adjust parameters "
                           "further")
                _logger.info(self.__class__.__name__ + ": " + message)
                return X.copy(), y.copy()
            else:
                return DBSMOTE(proportion=self.proportion,
                               eps=self.eps*1.5,
                               min_samples=self.min_samples-1,
                               nn_params=self.nn_params,
                               n_jobs=self.n_jobs,
                               random_state=self._random_state_init).sample(X, y)

        # determining cluster size distribution
        clusters = [np.where(labels == i)[0] for i in range(num_labels)]
        cluster_sizes = np.array([np.sum(labels == i)
                                  for i in range(num_labels)])
        cluster_dist = cluster_sizes/np.sum(cluster_sizes)

        # Bellman-Ford algorithm, inspired by
        # https://gist.github.com/joninvski/701720
        def initialize(graph, source):
            """
            Initializes shortest path algorithm.

            Args:
                graph (dict): graph in dictionary representation
                source (key): source node

            Returns:
                dict, dict: initialized distance and path dictionaries
            """
            d = {}
            p = {}
            for node in graph:
                d[node] = float('Inf')
                p[node] = None
            d[source] = 0
            return d, p

        def relax(u, v, graph, d, p):
            """
            Checks if shorter path exists.

            Args:
                u (key): key of a node
                v (key): key of another node
                graph (dict): the graph object
                d (dict): the distances dictionary
                p (dict): the paths dictionary
            """
            if d[v] > d[u] + graph[u][v]:
                d[v] = d[u] + graph[u][v]
                p[v] = u

        def bellman_ford(graph, source):
            """
            Main entry point of the Bellman-Ford algorithm

            Args:
                graph (dict): a graph in dictionary representation
                source (key): the key of the source node
            """
            d, p = initialize(graph, source)
            for i in range(len(graph)-1):
                for u in graph:
                    for v in graph[u]:
                        relax(u, v, graph, d, p)
            for u in graph:
                for v in graph[u]:
                    assert d[v] <= d[u] + graph[u][v]
            return d, p

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # extract graphs and center-like objects
        graphs = []
        centroid_indices = []
        shortest_paths = []
        for c in range(num_labels):
            # extracting the cluster elements
            cluster = X_min[clusters[c]]
            # initializing the graph object
            graph = {}
            for i in range(len(cluster)):
                graph[i] = {}

            # fitting nearest neighbors model to the cluster elements
            nn = NearestNeighborsWithMetricTensor(n_neighbors=len(cluster), 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
            nn.fit(cluster)
            dist, ind = nn.kneighbors(cluster)

            # extracting graph edges according to directly density reachabality
            # definition
            for i in range(len(cluster)):
                n = min([len(cluster), (self.min_samples + 1)])
                index_set = ind[i][1:n]
                for j in range(len(cluster)):
                    if j in index_set and dist[i][ind[i] == j][0] < self.eps:
                        graph[i][j] = dist[i][ind[i] == j][0]
            graphs.append(graph)
            # finding the index of the center like object
            centroid_ind = nn.kneighbors(
                np.mean(cluster, axis=0).reshape(1, -1), return_distance=False)[0][0]
            centroid_indices.append(centroid_ind)
            # extracting shortest paths from centroid object
            shortest_paths.append(bellman_ford(graph, centroid_ind))

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(
                np.arange(len(clusters)), p=cluster_dist)
            cluster = X_min[clusters[cluster_idx]]
            idx = self.random_state.choice(range(len(clusters[cluster_idx])))

            # executing shortest path algorithm
            distances, parents = shortest_paths[cluster_idx]

            # extracting path
            path = [idx]
            while not parents[path[-1]] is None:
                path.append(parents[path[-1]])

            if len(path) == 1:
                # if the center like object is selected
                samples.append(cluster[path[0]])
            elif len(path) == 2:
                # if the path consists of 1 edge
                X_a = cluster[path[0]]
                X_b = cluster[path[1]]
                sample = self.sample_between_points_componentwise(X_a, X_b)
                samples.append(sample)
            else:
                # if the path consists of at least two edges
                random_vertex = self.random_state.randint(len(path)-1)
                X_a = cluster[path[random_vertex]]
                X_b = cluster[path[random_vertex + 1]]
                sample = self.sample_between_points_componentwise(X_a, X_b)
                samples.append(sample)

        return (np.vstack([X, ss.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'eps': self.eps,
                'min_samples': self.min_samples,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
