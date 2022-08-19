"""
This module implements the CURE_SMOTE method.
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from ..base import coalesce_dict, coalesce
from ..base import pairwise_distances_mahalanobis

from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['CURE_SMOTE']

class CURE_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @Article{cure_smote,
                        author="Ma, Li
                        and Fan, Suohai",
                        title="CURE-SMOTE algorithm and hybrid algorithm for
                                feature selection and parameter optimization
                                based on random forests",
                        journal="BMC Bioinformatics",
                        year="2017",
                        month="Mar",
                        day="14",
                        volume="18",
                        number="1",
                        pages="169",
                        issn="1471-2105",
                        doi="10.1186/s12859-017-1578-z",
                        url="https://doi.org/10.1186/s12859-017-1578-z"
                        }

    Notes:
        * It is not specified how to determine the cluster with the
            "slowest growth rate"
        * All clusters can be removed as noise.
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_clusters=5,
                 noise_th=2,
                 nn_params=None,
                 ss_params=None,
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
            n_clusters (int): number of clusters to generate
            noise_th (int): below this number of elements the cluster is
                                considered as noise
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
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
        self.check_greater_or_equal(n_clusters, "n_clusters", 1)
        self.check_greater_or_equal(noise_th, "noise_th", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_clusters = n_clusters
        self.noise_th = noise_th
        self.nn_params = coalesce(nn_params, {})
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
                                  'n_clusters': [5, 10, 15],
                                  'noise_th': [1, 3]}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def do_the_clustering(self, X_min, nn_params):
        """
        Do the clustering

        Args:
            X_min (np.array): the minority samples
            nn_params (dict): nearest neighbors (distance) parameters

        Returns:
            list(np.array): the clusters
        """
        # initiating clustering
        clusters = [np.array([i]) for i in range(len(X_min))]
        distm = pairwise_distances_mahalanobis(X_min,
                                        tensor=nn_params.get('metric_tensor', None))

        # setting the diagonal of the distance matrix to infinity
        for idx in range(len(distm)):
            distm[idx, idx] = np.inf

        # starting the clustering iteration
        iteration = 0
        while len(clusters) > self.n_clusters:
            iteration = iteration + 1

            # delete a cluster with slowest growth rate, determined by
            # the cluster size
            if iteration % self.n_clusters == 0:
                # extracting cluster sizes
                cluster_sizes = np.array([len(c) for c in clusters])
                # removing one of the clusters with the smallest size
                to_remove = np.where(cluster_sizes == np.min(cluster_sizes))[0]
                to_remove = self.random_state.choice(to_remove)
                del clusters[to_remove]
                # adjusting the distance matrix accordingly
                distm = np.delete(distm, to_remove, axis=0)
                distm = np.delete(distm, to_remove, axis=1)

            # finding the cluster pair with the smallest distance
            min_coord = np.where(distm == np.min(distm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # merging the clusters
            clusters[merge_a] = np.hstack(
                [clusters[merge_a], clusters[merge_b]])
            # removing one of them
            del clusters[merge_b]
            # adjusting the distances in the distance matrix
            distm[merge_a] = np.min(np.vstack([distm[merge_a], distm[merge_b]]), axis=0)
            distm[:, merge_a] = distm[merge_a]
            # removing the row and column corresponding to one of
            # the merged clusters
            distm = np.delete(distm, merge_b, axis=0)
            distm = np.delete(distm, merge_b, axis=1)
            # updating the diagonal
            for idx in range(len(distm)):
                distm[idx, idx] = np.inf

        # removing clusters declared as noise
        to_remove = []
        for idx, cluster in enumerate(clusters):
            if len(cluster) < self.noise_th:
                to_remove.append(idx)
        clusters = [clusters[i]
                    for i in range(len(clusters)) if i not in to_remove]

        return clusters

    def generate_samples_in_clusters(self, X_min, clusters, n_to_sample):
        """
        Generate samples within clusters

        Args:
            X_min (np.array): the minority samples
            clusters (list(np.array)): the cluster indices

        Returns:
            np.array: the generated samples
        """
        clusters_selected = self.random_state.choice(len(clusters), n_to_sample)
        cluster_unique, cluster_count = np.unique(clusters_selected,
                                                    return_counts=True)

        samples = []

        for idx, cluster in enumerate(cluster_unique):
            cluster_vectors = X_min[clusters[cluster]]
            cluster_mean = np.mean(cluster_vectors, axis=0)

            #if len(clusters[idx]) >= self.n_dim:
            X = np.array([cluster_mean])
            indices = np.array([np.hstack([np.array([0]),
                                np.arange(len(cluster_vectors))])])
            samples.append(self.sample_simplex(X=X,
                                                indices=indices,
                                                n_to_sample=cluster_count[idx],
                                                X_vertices=cluster_vectors))
            #else:
            #    sample_indices = self.random_state.choice(len(clusters[cluster]),
            #                                                cluster_count[idx])
            #    samples.append(cluster_vectors[sample_indices])

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

        # standardizing the data
        mms = MinMaxScaler()
        X_scaled = mms.fit_transform(X) # pylint: disable=invalid-name

        X_min = X_scaled[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        clusters = self.do_the_clustering(X_min, nn_params)

        # all clusters can be noise
        if len(clusters) == 0:
            return self.return_copies(X, y, "all clusters are removed as noise")

        samples = self.generate_samples_in_clusters(X_min,
                                                    clusters,
                                                    n_to_sample)

        return (np.vstack([X, mms.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_clusters': self.n_clusters,
                'noise_th': self.noise_th,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
