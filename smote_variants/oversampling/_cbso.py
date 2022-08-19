"""
This module implements the CBSO method.
"""

import numpy as np
from scipy.linalg import circulant

from ..base import coalesce_dict, fix_density, coalesce
from ..base import (NearestNeighborsWithMetricTensor,
                             pairwise_distances_mahalanobis)
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['CBSO']

class CBSO(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{cbso,
                            author="Barua, Sukarna
                            and Islam, Md. Monirul
                            and Murase, Kazuyuki",
                            editor="Lu, Bao-Liang
                            and Zhang, Liqing
                            and Kwok, James",
                            title="A Novel Synthetic Minority Oversampling
                                    Technique for Imbalanced Data Set
                                    Learning",
                            booktitle="Neural Information Processing",
                            year="2011",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="735--744",
                            isbn="978-3-642-24958-7"
                            }

    Notes:
        * Clusters containing 1 element induce cloning of samples.
    """

    categories = [OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_density_based,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 C_p=1.3,
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
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): the simplex sampling parameters
            C_p (float): used to set the threshold of clustering
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
        self.check_greater(C_p, "C_p", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.C_p = C_p # pylint: disable=invalid-name
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
                                  'C_p': [0.8, 1.0, 1.3, 1.6]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def do_clustering(self, X_min, nn_params):
        """
        Do the clustering of minority samples.

        Args:
            X_min (np.array): minority samples
            nn_params (dict): nearest neighbor parameters

        Returns:
            np.array, np.array: clusters, labels
        """
        # do the clustering
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=2,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        d_avg = np.mean(nnmt.kneighbors(X_min)[0][:, 1])
        T_h = d_avg * self.C_p # pylint: disable=invalid-name

        # initiating clustering
        clusters = [np.array([i]) for i in range(len(X_min))]
        distm = pairwise_distances_mahalanobis(X_min,
                                            tensor=nn_params.get('metric_tensor', None))

        # setting the diagonal of the distance matrix to infinity
        for idx in range(len(distm)):
            distm[idx, idx] = np.inf

        # starting the clustering iteration
        while True:
            # finding the cluster pair with the smallest distance
            min_coord = np.where(distm == np.min(distm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # check termination conditions
            if distm[merge_a, merge_b] > T_h or len(distm) == 1:
                break

            # merging the clusters
            clusters[merge_a] = np.hstack(
                [clusters[merge_a], clusters[merge_b]])
            # removing one of them
            del clusters[merge_b]
            # adjusting the distances in the distance matrix
            distm[merge_a] = np.min(np.vstack([distm[merge_a], distm[merge_b]]), axis=0)
            distm[:, merge_a] = distm[merge_a]
            # removing the row and column corresponding to one of the
            # merged clusters
            distm = np.delete(distm, merge_b, axis=0)
            distm = np.delete(distm, merge_b, axis=1)
            # updating the diagonal
            for idx in range(len(distm)):
                distm[idx, idx] = np.inf

        # extracting cluster labels
        labels = np.zeros(len(X_min)).astype(int)
        for idx, cluster in enumerate(clusters):
            for jdx in cluster:
                labels[jdx] = idx

        return clusters, labels

    def generate_samples_in_clusters(self,
                                    *,
                                    clusters,
                                    X_min,
                                    n_to_sample,
                                    weights):
        """
        Generate samples within the clusters.

        Args:
            clusters (np.array): the clusters
            X_min (np.array): the minority samples
            n_to_sample (int): number of samples to generate
            weights (np.array): the minority point density

        Returns:
            np.array: the generated samples
        """

        cluster_weights = np.array([np.sum(weights[cluster]) for cluster in clusters])
        cluster_weights = cluster_weights / np.sum(cluster_weights)

        cluster_indices = self.random_state.choice(np.arange(len(clusters)),
                                                    n_to_sample,
                                                    p=cluster_weights)

        cluster_unique, cluster_count = np.unique(cluster_indices,
                                                    return_counts=True)

        samples = []
        for idx, cluster in enumerate(cluster_unique):
            cluster_vectors = X_min[clusters[cluster]]
            within_cluster_weights = weights[clusters[cluster]]
            within_cluster_weights = fix_density(within_cluster_weights)

            #if len(cluster_vectors) >= self.n_dim:
            samples.append(self.sample_simplex(X=cluster_vectors,
                                        indices=circulant(np.arange(cluster_vectors.shape[0])),
                                        n_to_sample=cluster_count[idx],
                                        base_weights=within_cluster_weights))
            #else:
            #    sample_indices = self.random_state.choice(np.arange(len(cluster_vectors)),
            #                                                cluster_count[idx],
            #                                                p=within_cluster_weights)
            #    samples.append(cluster_vectors[sample_indices])

        # original implementation
        # do the sampling
        #samples = []
        #while len(samples) < n_to_sample:
        #    idx = self.random_state.choice(np.arange(len(X_min)), p=weights)
        #    if len(clusters[labels[idx]]) <= 1:
        #        samples.append(X_min[idx])
        #        continue
        #
        #    random_idx = self.random_state.choice(clusters[labels[idx]])
        #    while random_idx == idx:
        #        random_idx = self.random_state.choice(
        #            clusters[labels[idx]])
        #    samples.append(self.sample_between_points(
        #        X_min[idx], X_min[random_idx]))

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

        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params,
                                                                        X, y)

        # fitting nearest neighbors model to find neighbors of minority points
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=self.n_neighbors + 1,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        # extracting the number of majority neighbors
        weights = [np.sum(y[ind[i][1:]] == self.maj_label)
                   for i in range(len(X_min))]

        weights = fix_density(weights)

        clusters, _ = self.do_clustering(X_min, nn_params)

        samples = self.generate_samples_in_clusters(clusters=clusters,
                                                    n_to_sample=n_to_sample,
                                                    X_min=X_min,
                                                    weights=weights)

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
                'C_p': self.C_p,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
