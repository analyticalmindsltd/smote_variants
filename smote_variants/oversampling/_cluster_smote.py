"""
This module implements the cluster_SMOTE method.
"""
import warnings

import numpy as np

from sklearn.cluster import KMeans

from ..config import suppress_external_warnings
from ..base import coalesce_dict, coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['cluster_SMOTE']

class cluster_SMOTE(OverSamplingSimplex): # pylint: disable=invalid-name
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=3,
                 *,
                 nn_params=None,
                 ss_params=None,
                 n_clusters=3,
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
            n_neighbors (int): number of neighbors in SMOTE
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): the simplex sampling parameters
            n_clusters (int): number of clusters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_components", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.n_clusters = n_clusters
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
                                  'n_neighbors': [3, 5, 7],
                                  'n_clusters': [3, 5, 7, 9]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_clusters_and_neighbood_graph(self,
                                                cluster_indices,
                                                X_min,
                                                nn_params):
        """
        Determine clusters and neighborhood graphs.

        Args:
            cluster_indices (np.array): the cluster labeling
            X_min (np.array): the vectors
            nn_params (dict): nearest neighbors parameters

        Returns:
            np.array, np.array: the clusters and the neighborhood graphs
        """
        cluster_vectors = [X_min[cluster_indices[idx]] \
                                for idx in range(len(cluster_indices))]
        cluster_nn_indices = []

        for idx, cluster in enumerate(cluster_vectors):
            n_neighbors = min([self.n_neighbors, len(cluster_indices[idx])])
            nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
            nnmt.fit(cluster)
            cluster_nn_indices.append(nnmt.kneighbors(cluster,
                                                        return_distance=False))

        return cluster_vectors, cluster_nn_indices

    def generate_samples_in_clusters(self,
                                    cluster_indices,
                                    X_min,
                                    nn_params,
                                    n_to_sample):
        """
        Generate samples within the clusters.

        Args:
            cluster_indices (np.array): the cluster labeling
            X_min (np.array): the vectors
            nn_params (dict): the nearest neighbors parameters
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        clusters, indices = \
            self.determine_clusters_and_neighbood_graph(cluster_indices,
                                                        X_min,
                                                        nn_params)

        clusters_selected = self.random_state.choice(len(clusters),
                                                        n_to_sample)
        cluster_unique, cluster_count = np.unique(clusters_selected,
                                                    return_counts=True)

        samples = []
        for idx, cluster in enumerate(cluster_unique):
            #if len(clusters[cluster]) >= self.n_dim:
            samples.append(self.sample_simplex(X=clusters[cluster],
                                        indices=indices[cluster],
                                        n_to_sample=cluster_count[idx]))
            #else:
            #    sample_indices = self.random_state.choice(len(clusters[cluster]),
            #                                                cluster_count[idx])
            #    samples.append(clusters[cluster][sample_indices])

        return np.vstack(samples)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        kmeans = KMeans(n_clusters=min([len(X_min), self.n_clusters]),
                        random_state=self._random_state_init)
        with warnings.catch_warnings():
            if suppress_external_warnings():
                warnings.simplefilter("ignore")
            kmeans.fit(X_min)

        unique_labels = np.unique(kmeans.labels_)

        # creating nearest neighbors objects for each cluster
        cluster_indices = [np.where(kmeans.labels_ == c)[0]
                           for c in unique_labels]

        if np.max([len(cluster) for cluster in cluster_indices]) <= 1:
            return self.return_copies(X, y, "All clusters contain 1 element")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        samples = self.generate_samples_in_clusters(cluster_indices,
                                                    X_min,
                                                    nn_params,
                                                    n_to_sample)

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
                'n_clusters': self.n_clusters,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
