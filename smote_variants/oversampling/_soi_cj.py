"""
This module implements the SOI_CJ method.
"""
import itertools

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import (NearestNeighborsWithMetricTensor,
                            pairwise_distances_mahalanobis)
from ..base import OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['SOI_CJ']

class SOI_CJ(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_sample_componentwise,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 method='jittering',
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
        ss_params_default = {'n_dim': 1, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_isin(method, 'method', ['interpolation', 'jittering'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.method = method
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
                                  'method': ['interpolation', 'jittering']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def rearrange_clusters(self, cluster_idx, cluster_jdx, intersection):
        """
        Move the difference into the bigger cluster.

        Args:
            cluster_idx (set): the first cluster
            cluster_jdx (set): the second cluster
            intersection (set): the intersection of the clusters

        Returns:
            set, set: the updated cluster_idx and cluster_jdx
        """
        # otherwise move the difference into the
        # bigger cluster
        if len(cluster_idx) > len(cluster_jdx):
            # removes the items that exist in both sets
            cluster_jdx.difference_update(intersection)
        else:
            cluster_idx.difference_update(intersection)

        return cluster_idx, cluster_jdx

    def update_clusters(self,
                            *,
                            X,
                            y,
                            cluster_idx,
                            cluster_jdx,
                            intersection,
                            nn_all,
                            nn_params):
        """
        Update two clusters during the merging.

        Args:
            X (np.array): the training vectors
            y (np.array): the target labels
            cluster_idx (set): the cluster at the index idx
            cluster_jdx (set): the cluster at the index jdx
            intersection (set): the intersection of the clusters
            nn_all (obj): fitted nearest neighbors object
            nn_params (dict): nearest neighbor parameters

        Returns:
            set, set: the updated clusters cluster_idx and cluster_jdx
        """
        # computing distance matrix
        distm = pairwise_distances_mahalanobis(X[list(cluster_idx)],
                                                Y=X[list(cluster_jdx)],
                                                tensor=nn_params.get('metric_tensor', None))
        # largest distance
        max_dist_pair = np.where(distm == np.max(distm))
        # elements with the largest distance
        max_i = X[list(cluster_idx)[max_dist_pair[0][0]]]
        max_j = X[list(cluster_jdx)[max_dist_pair[1][0]]]

        # finding midpoint and radius
        mid_point = (max_i + max_j)/2.0
        radius = np.linalg.norm(mid_point - max_i)

        # extracting points within the hypersphare of
        # radius "radius"
        ind = nn_all.radius_neighbors(mid_point.reshape(1, -1),
                                        radius,
                                        return_distance=False)

        if np.sum(y[ind[0].astype(int)] == self.min_label) > len(ind[0])/2:
            # if most of the covered elements come from the
            # minority class, merge clusters
            cluster_idx.update(cluster_jdx)
            cluster_jdx = set()
        else:
            cluster_idx, cluster_jdx = \
                self.rearrange_clusters(cluster_idx, cluster_jdx, intersection)

        return cluster_idx, cluster_jdx

    def init_clusters(self, y, indices):
        """
        Initialize the clusters.

        Args:
            y (np.array): all target labels
            indices (np.array): the neighborhood structure

        Returns:
            list(set): the initial clusters
        """
        clusters = []

        first_maj_index = np.argmax(y[indices] != self.min_label, axis=1)

        for idx in range(indices.shape[0]):
            # while the closest instance is from the minority class, adding it
            # to the cluster
            clusters.append(set(indices[idx, :first_maj_index[idx]].tolist()))

        return clusters

    def clustering(self, X, y, nn_params):
        """
        Implementation of the clustering technique described in the paper.

        Args:
            X (np.array): array of training instances
            y (np.array): target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            list(set): list of minority clusters
        """
        nn_all= NearestNeighborsWithMetricTensor(n_jobs=self.n_jobs,
                                                 **nn_params)
        nn_all.fit(X)

        X_min = X[y == self.min_label]

        # extract nearest neighbors of all samples from the set of
        # minority samples
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=len(X_min),
                                               n_jobs=self.n_jobs,
                                               **nn_params)
        nnmt.fit(X)
        indices = nnmt.kneighbors(X_min, return_distance=False)

        # initialize clusters by minority samples
        clusters = self.init_clusters(y, indices)

        # cluster merging phase
        is_intersection = True
        while is_intersection:
            is_intersection = False
            for idx, cluster_idx in enumerate(clusters):
                for jdx, cluster_jdx in enumerate(clusters[(idx + 1):]):
                    # computing intersection
                    intersection = cluster_idx.intersection(cluster_jdx)
                    if len(intersection) > 0:
                        is_intersection = True

                        cluster_idx, cluster_jdx = self.update_clusters(X=X, y=y,
                                                            cluster_idx=cluster_idx,
                                                            cluster_jdx=cluster_jdx,
                                                            intersection=intersection,
                                                            nn_all=nn_all,
                                                            nn_params=nn_params)

                        clusters[idx] = cluster_idx
                        clusters[jdx] = cluster_jdx


            clusters = [c for c in clusters if len(c) > 0]

        # returning non-empty clusters
        return clusters

    def sample_within_cluster_interpolation(self, X, cluster, n_to_sample):
        """
        Sample within a cluster by interpolation.

        Args:
            X (np.array): all training vectors
            cluster (set): the cluster indices
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        combinations = list(itertools.combinations(np.arange(len(cluster)), 2))
        indices = self.random_state.choice(np.arange(len(combinations)),
                                                        n_to_sample)

        combinations = np.array(combinations)[indices]
        coords = np.array(list(zip(*combinations)))

        X_0 = X[np.array(cluster)[coords[0]]] # pylint: disable=invalid-name
        X_1 = X[np.array(cluster)[coords[1]]] # pylint: disable=invalid-name
        samples = X_0 + (X_1 - X_0) \
                        * self.random_state.random_sample(size=X_0.shape)

        return samples

    def sampling_by_interpolation(self,
                                X,
                                clusters,
                                cluster_weights,
                                n_to_sample):
        """
        Generate samples by interpolation.

        Args:
            X (np.array): all training vectors
            clusters (list(set)): the cluster indices
            cluster_weights (np.array): the weights of the clusters
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        samples = np.zeros(shape=(0, X.shape[1]))
        cluster_indices = self.random_state.choice(np.arange(len(clusters)),
                                                    n_to_sample,
                                                    p=cluster_weights)
        cluster_unique, cluster_count = np.unique(cluster_indices,
                                                    return_counts=True)
        for idx, cluster_idx in enumerate(cluster_unique):
            samples_tmp = self.sample_within_cluster_interpolation(X,
                                                clusters[cluster_idx],
                                                cluster_count[idx])
            samples = np.vstack([samples, samples_tmp])

        return samples

    def cluster_stds(self, X, y, clusters):
        """
        Determine the cluster dependent stds.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            clusters (list(set)): the cluster indices

        Returns:
            np.array: the stds for the clusters
        """
        std_min = np.std(X[y == self.min_label], axis=0)
        cluster_stds = []

        for cluster in clusters:
            std = np.std(X[cluster], axis=0)
            std = np.min(np.vstack([std_min, std]), axis=0)
            cluster_stds.append(std)

        return np.array(cluster_stds)

    def sample_within_cluster_jittering(self,
                                        *,
                                        X,
                                        cluster,
                                        n_to_sample,
                                        std,
                                        nn_params):
        """
        Sample within a cluster by interpolation.

        Args:
            X (np.array): all training vectors
            cluster (set): the cluster indices
            n_to_sample (int): the number of samples to generate
            std (np.array): the cluster specific standard deviations

        Returns:
            np.array: the generated samples
        """

        self.gaussian_component = {'sigmas': std}

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=len(cluster),
                                               n_jobs=self.n_jobs,
                                               **nn_params)
        nnmt.fit(X[cluster])
        indices = nnmt.kneighbors(X[cluster], return_distance=False)

        samples = self.sample_simplex(X=X[cluster],
                                      indices=indices,
                                      n_to_sample=n_to_sample)

        #indices = self.random_state.choice(np.arange(len(cluster)),
        #                                        n_to_sample)
        #X_base = X[np.array(cluster)[indices]]
        #samples = X_base + self.random_state.normal(size=X_base.shape) * std

        return samples

    def sampling_by_jittering(self, *, X, y,
                            clusters,
                            cluster_weights,
                            n_to_sample,
                            nn_params):
        """
        Generate samples by jittering.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            clusters (list(set)): the cluster indices
            cluster_weights (np.array): the weights of the clusters
            n_to_sample (int): the number of samples to generate
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the generated samples
        """
        stds = self.cluster_stds(X, y, clusters)

        samples = np.zeros(shape=(0, X.shape[1]))

        cluster_indices = self.random_state.choice(np.arange(len(clusters)),
                                                    n_to_sample,
                                                    p=cluster_weights)
        cluster_unique, cluster_count = np.unique(cluster_indices,
                                                    return_counts=True)

        for idx, cluster_idx in enumerate(cluster_unique):
            X_samp = self.sample_within_cluster_jittering(X=X,
                                                cluster=clusters[cluster_idx],
                                                n_to_sample=cluster_count[idx],
                                                std=stds[cluster_idx],
                                                nn_params=nn_params)
            samples = np.vstack([samples, X_samp])

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
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        # do the clustering
        _logger.info("%s: Executing clustering", self.__class__.__name__)
        clusters = self.clustering(X, y, nn_params)

        # filtering the clusters, at least two points in a cluster are needed
        # for both interpolation and jittering (due to the standard deviation)
        clusters_filtered = [list(c) for c in clusters if len(c) > 2]

        if len(clusters_filtered) == 0:
            return self.return_copies(X, y, "No clusters with at least 2 elements")

        # if there are clusters having at least 2 elements, do the sampling
        cluster_nums = [len(c) for c in clusters_filtered]
        cluster_weights = cluster_nums/np.sum(cluster_nums)

        _logger.info("%s: Executing sample generation", self.__class__.__name__)

        if self.method == 'interpolation':
            samples = self.sampling_by_interpolation(X,
                                                clusters_filtered,
                                                cluster_weights,
                                                n_to_sample)
        elif self.method == 'jittering':
            samples = self.sampling_by_jittering(X=X,
                                                 y=y,
                                                 clusters=clusters_filtered,
                                                 cluster_weights=cluster_weights,
                                                 n_to_sample=n_to_sample,
                                                 nn_params=nn_params)

        return (np.vstack([X, samples]),
                np.hstack([y, np.array([self.min_label]*len(samples))]))

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
                **OverSamplingSimplex.get_params(self)}
