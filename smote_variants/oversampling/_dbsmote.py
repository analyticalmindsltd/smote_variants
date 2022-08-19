"""
This module implements the DBSMOTE method.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['DBSMOTE']

DB_LIMIT = 5

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
    distance = {}
    path = {}
    for node in graph:
        distance[node] = float('Inf')
        path[node] = None
    distance[source] = 0
    return distance, path

def relax(u_key, v_key, graph, distance, path):
    """
    Checks if shorter path exists.

    Args:
        u (key): key of a node
        v (key): key of another node
        graph (dict): the graph object
        distance (dict): the distances dictionary
        path (dict): the paths dictionary
    """
    if distance[v_key] > distance[u_key] + graph[u_key][v_key]:
        distance[v_key] = distance[u_key] + graph[u_key][v_key]
        path[v_key] = u_key

def bellman_ford(graph, source):
    """
    Main entry point of the Bellman-Ford algorithm

    Args:
        graph (dict): a graph in dictionary representation
        source (key): the key of the source node
    """
    distance, path = initialize(graph, source)
    for _ in range(len(graph)-1):
        for u_key in graph:
            for v_key in graph[u_key]:
                relax(u_key, v_key, graph, distance, path)
    for u_key in graph:
        for v_key in graph[u_key]:
            assert distance[v_key] <= distance[u_key] + graph[u_key][v_key]
    return distance, path


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
                 db_iter_limit=3,
                 nn_params={},
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            eps (float): eps parameter of DBSCAN
            min_samples (int): min_samples parameter of DBSCAN
            db_iter_limit (int): DBSCAN iterations limit
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater(eps, "eps", 0)
        self.check_greater_or_equal(min_samples, "min_samples", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.eps = eps
        self.min_samples = min_samples
        self.db_iter_limit = db_iter_limit
        self.nn_params = nn_params
        self.n_jobs = n_jobs

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

    def execute_dbscan(self, X_min):
        """
        Execute DBSCAN by releasing thresholds

        Args:
            X_min (np.array): the minority samples

        Returns:
            np.array: the labels
        """
        dbscan = DBSCAN(eps=self.eps,
                    min_samples=self.min_samples,
                    n_jobs=self.n_jobs).fit(X_min)
        labels = dbscan.labels_
        num_labels = np.max(labels) + 1

        step = 1

        while num_labels == 0 and step < self.db_iter_limit:
            # adjusting the parameters if no clusters were identified
            dbscan = DBSCAN(eps=self.eps*1.5**step,
                    min_samples=self.min_samples - 1*step,
                    n_jobs=self.n_jobs).fit(X_min)
            labels = dbscan.labels_
            num_labels = np.max(labels) + 1
            step = step + 1

        return labels

    def extract_shortest_paths(self,
                                X_min,
                                clusters,
                                nn_params):
        """
        Extract shortest paths.

        Args:
            X_min (np.array): minority samples
            clusters (np.array): clusters
            nn_params (dict): nearest neighbors params

        Returns:
            list: shortest paths
        """
        shortest_paths = []
        for _, clust in enumerate(clusters):
            # extracting the cluster elements
            cluster = X_min[clust]
            # initializing the graph object
            graph = {}
            for idx in range(len(cluster)):
                graph[idx] = {}

            # fitting nearest neighbors model to the cluster elements
            nnmt = NearestNeighborsWithMetricTensor(n_neighbors=len(cluster),
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
            nnmt.fit(cluster)
            dist, ind = nnmt.kneighbors(cluster)

            # extracting graph edges according directly to the density reachabality
            # definition
            for idx in range(len(cluster)):
                index_set = ind[idx][1:min([len(cluster),
                                            (self.min_samples + 1)])]
                for jdx in range(len(cluster)):
                    if jdx in index_set \
                        and dist[idx][ind[idx] == jdx][0] < self.eps:
                        graph[idx][jdx] = dist[idx][ind[idx] == jdx][0]

            # finding the index of the center like object
            centroid_ind = nnmt.kneighbors(np.mean(cluster, axis=0).reshape(1, -1),
                                           return_distance=False)[0][0]

            # extracting shortest paths from centroid object
            shortest_paths.append(bellman_ford(graph, centroid_ind))

        return shortest_paths

    def generate_samples(self,
                            n_to_sample,
                            clusters,
                            shortest_paths,
                            X_min):
        """
        Generate samples

        Args:
            n_to_sample (int): number of samples to be generates
            clusters (list): clusters
            shortest_paths (list): the shortest paths
            X_min (np.array): the minority samples

        Returns:
            np.array: the generated samples
        """
        cluster_dist = np.array([len(clusters[idx])
                                  for idx in range(len(clusters))])
        cluster_dist = cluster_dist/np.sum(cluster_dist)

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(
                np.arange(len(clusters)), p=cluster_dist)
            cluster = X_min[clusters[cluster_idx]]
            idx = self.random_state.choice(range(len(clusters[cluster_idx])))

            # executing shortest path algorithm
            _, parents = shortest_paths[cluster_idx]

            # extracting path
            path = [idx]
            while not parents[path[-1]] is None:
                path.append(parents[path[-1]])

            if len(path) == 1:
                # if the center like object is selected
                samples.append(cluster[path[0]])
            elif len(path) == 2:
                # if the path consists of 1 edge
                sample = \
                    self.sample_between_points_componentwise(cluster[path[0]],
                                                             cluster[path[1]])
                samples.append(sample)
            else:
                # if the path consists of at least two edges
                r_vertex = self.random_state.randint(len(path)-1)
                sample = \
                    self.sample_between_points_componentwise(cluster[path[r_vertex]],
                                                             cluster[path[r_vertex + 1]])
                samples.append(sample)
        return samples

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        scaler = StandardScaler().fit(X)
        X_ss = scaler.transform(X) # pylint: disable=invalid-name

        # doing the clustering using DBSCAN
        X_min = X_ss[y == self.min_label]
        labels = self.execute_dbscan(X_min)
        num_labels = np.max(labels) + 1

        if num_labels == 0:
            return self.return_copies(X, y, "no reasonable clusters found")

        # determining cluster size distribution
        clusters = [np.where(labels == i)[0] for i in range(num_labels)]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        shortest_paths = self.extract_shortest_paths(X_min, clusters, nn_params)

        samples = self.generate_samples(n_to_sample, clusters, shortest_paths,
                            X_min)

        return (np.vstack([X, scaler.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'eps': self.eps,
                'min_samples': self.min_samples,
                'db_iter_limit': self.db_iter_limit,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
