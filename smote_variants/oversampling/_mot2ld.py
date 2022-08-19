"""
This module implements the MOT2LD method.
"""

import warnings

import numpy as np

from sklearn.manifold import TSNE
import scipy.signal as ssignal
from scipy.linalg import circulant

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['MOT2LD']

class MOT2LD(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{mot2ld,
                            author="Xie, Zhipeng
                            and Jiang, Liyang
                            and Ye, Tengju
                            and Li, Xiaoli",
                            editor="Renz, Matthias
                            and Shahabi, Cyrus
                            and Zhou, Xiaofang
                            and Cheema, Muhammad Aamir",
                            title="A Synthetic Minority Oversampling Method
                                    Based on Local Densities in Low-Dimensional
                                    Space for Imbalanced Learning",
                            booktitle="Database Systems for Advanced
                                        Applications",
                            year="2015",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="3--18",
                            isbn="978-3-319-18123-3"
                            }

    Notes:
        * Clusters might contain 1 elements, and all points can be filtered
            as noise.
        * Clusters might contain 0 elements as well, if all points are filtered
            as noise.
        * The entire clustering can become empty.
        * TSNE is very slow when the number of instances is over a couple
            of 1000
    """

    categories = [OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_components=2,
                 k=5,
                 nn_params=None,
                 ss_params=None,
                 d_cut='auto',
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
            n_components (int): number of components for stochastic
                                neighborhood embedding
            k (int): number of neighbors in the nearest neighbor component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            d_cut (float/str): distance cut value/'auto' for automated
                                selection
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_components, 'n_component', 1)
        self.check_greater_or_equal(k, 'k', 1)
        if isinstance(d_cut, (int, float)):
            if d_cut <= 0:
                raise ValueError(f"{self.__class__.__name__}: Non-positive "\
                                        "d_cut is not allowed")
        elif d_cut != 'auto':
            raise ValueError(f"{self.__class__.__name__}: d_cut value "\
                                f"{d_cut} not implemented")
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.k = k # pylint: disable=invalid-name
        self.nn_params = nn_params
        self.d_cut = d_cut
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
                                  'n_components': [2],
                                  'k': [3, 5, 7],
                                  'd_cut': ['auto']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def density_peak_clustering(self, rho, indices_min, distances_min):
        """
        Do density peak clustering.

        Args:
            rho (np.array): the rho array
            indices_min (np.array): the minority neighborhood structure
            distances_min (np.array): the minority neighbor distances

        Returns:
            np.array, np.array, np.array: the r, d and idx arrays
        """
        # extracting the number of neighbors in a given radius
        closest_highest = []
        delta = []

        # implementation of the density peak clustering algorithm
        # based on http://science.sciencemag.org/content/344/6191/1492.full
        for idx, rho_val in enumerate(rho):
            closest_neighbors = indices_min[idx]
            closest_densities = rho[closest_neighbors]
            closest_highs = np.where(closest_densities > rho_val)[0]

            if len(closest_highs) > 0:
                closest_highest.append(closest_highs[0])
                delta.append(distances_min[idx][closest_highs[0]])
            else:
                closest_highest.append(-1)
                delta.append(np.max(distances_min))

        to_sort = zip(rho, delta, np.arange(len(rho)))
        r, d, idx = zip(*sorted(to_sort, key=lambda x: x[0])) # pylint: disable=invalid-name
        r, d, idx = np.array(r), np.array(d), np.array(idx) # pylint: disable=invalid-name

        return r, d, idx

    def check_enough_clusters(self,
                                d # pylint: disable=invalid-name
                                ):
        """
        Check if there are enough clusters.

        Args:
            d (np.array): clusters
        """
        if len(d) < 3:
            raise ValueError(f"Not enough clusters: {len(d)}")

    def check_enough_peaks(self, peak_indices):
        """
        Check if there are enough peaks.

        Args:
            peak_indices (np.array): the peak indices
        """
        if len(peak_indices) == 0:
            raise ValueError("No peaks found")

    def determine_cluster_centers(self, X_min, rho, indices_min, distances_min):
        """
        Determine the cluster centers.

        Args:
            X_min (np.array): the minority samples
            rho (np.array): the rho array
            indices_min (np.array): the minority neighborhood structure
            distances_min (np.array): the minority distances

        Returns:
            np.array: cluster centers
        """
        r, d, idx = self.density_peak_clustering(rho, # pylint: disable=invalid-name
                                        indices_min,
                                        distances_min)

        self.check_enough_clusters(d)

        widths = np.arange(1, int(np.rint(len(r)/2)))
        peak_indices = np.array(ssignal.find_peaks_cwt(d, widths=widths))

        self.check_enough_peaks(peak_indices)

        cluster_centers = X_min[idx[peak_indices]]

        return cluster_centers

    def determine_neighborhoods(self,
                                X_tsne, # pylint: disable=invalid-name
                                X_min,
                                nn_params):
        """
        Determine the neighborhood structures.

        Args:
            X_tsne (np.array): the transformed samples
            X_min (np.array): the minority samples
            nn_params (dict): the neighborhood parameters

        Returns:
            np.array, np.array, np.array, np.array:
                        the indices, minority distances,
                        indices and the rho array
        """
         # fitting nearest neighbors model for all training data
        n_neighbors = min([len(X_min), self.k + 1])
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_tsne)
        distances, indices = nnmt.kneighbors(X_min)

        # fitting nearest neighbors model to the minority data
        nn_min= NearestNeighborsWithMetricTensor(n_neighbors=len(X_min),
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nn_min.fit(X_min)
        distances_min, indices_min = nn_min.kneighbors(X_min)

        if isinstance(self.d_cut, (int, float)):
            d_cut = self.d_cut
        elif self.d_cut == 'auto':
            d_cut = np.max(distances[:, 1])

        rho = nnmt.radius_neighbors(X_min, d_cut, return_distance=False)
        rho = np.array([rho_row.shape[0] for rho_row in rho])

        return indices, distances_min, indices_min, rho

    def determine_clusters(self,
                            X_tsne, # pylint: disable=invalid-name
                            y,
                            X_min,
                            nn_params):
        """
        Do the clustering.

        Args:
            X_tsne (np.array): all samples after TSNE transformation
            y (np.array): target labels
            X_min (np.array): the minority labels after transformation
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array, np.array, np.array: cluster centers, noise mask
                                        and importance scores
        """
        indices, distances_min, indices_min, rho = \
                self.determine_neighborhoods(X_tsne, X_min, nn_params)

        cluster_centers = self.determine_cluster_centers(X_min,
                                                        rho,
                                                        indices_min,
                                                        distances_min)

        # computing local minority counts and determining noisy samples
        local_minority_count = np.sum(y[indices[:, 1:]] == self.min_label, axis=1)

        noise = np.where(np.logical_or(rho == 1, local_minority_count == 0))[0]

        return cluster_centers, noise, local_minority_count/rho

    def check_empty_clustering(self, cluster_indices):
        """
        Check if the clustering is empty

        Args:
            cluster_indices (np.array): cluster indices
        """
        # checking if clustering is empty
        empty_clustering = np.all([len(cluster) == 0 for cluster in cluster_indices])

        if empty_clustering:
            raise ValueError("Empty clustering")

    def check_probabilities(self, prob):
        """
        Check if the probabilities are empty.

        Args:
            prob (np.array): the probabilities
        """

        if np.sum(prob) == 0.0:
            raise ValueError("Empty probabilities.")

    def do_clustering(self,
                        X_tsne, # pylint: disable=invalid-name
                        y,
                        X_min):
        """
        Do the clustering.

        Args:
            X_tsne (np.array): all samples after TSNE transformation
            y (np.array): all target labels
            X_min (np.array): the minority labels after transformation

        Returns:
            np.array, np.array, np.array, np.array: the cluster indices,
                            cluster labels, probabilities and noise mask
        """
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                self.metric_tensor_from_nn_params(nn_params, X_tsne, y)

        cluster_centers, noise, importance = \
                self.determine_clusters(X_tsne, y, X_min, nn_params)

        # finding closest cluster center to minority points and deriving
        # cluster labels
        nn_cluster= NearestNeighborsWithMetricTensor(n_neighbors=1,
                                                        n_jobs=self.n_jobs,
                                                        **(nn_params))
        nn_cluster.fit(cluster_centers)
        ind_cluster = nn_cluster.kneighbors(X_min, return_distance=False)
        cluster_labels = ind_cluster[:, 0]

        prob = importance
        prob[noise] = 0.0

        self.check_probabilities(prob)

        prob = prob/np.sum(prob)

        # extracting cluster indices
        cluster_indices = [np.where(cluster_labels == i)[0]
                           for i in range(np.max(cluster_labels) + 1)]
        # removing noise from clusters
        cluster_indices = [list(set(c).difference(set(noise)))
                           for c in cluster_indices]

        self.check_empty_clustering(cluster_indices)

        cluster_sizes = np.array([len(cluster) for cluster in cluster_indices])
        empty = np.isin(cluster_labels, np.where(cluster_sizes == 0)[0])
        prob[empty] = 0.0
        prob = prob/np.sum(prob)

        return cluster_indices, prob, noise

    def determine_cluster_probabilities(self, cluster_indices, prob):
        """
        Determine the cluster probabilities.

        Args:
            cluster_indices (np.array): the cluster indices
            prob (np.array): the sample probabilities

        Returns:
            np.array: the cluster probabilities
        """
        cluster_probs = np.zeros(len(cluster_indices))
        for idx, indices in enumerate(cluster_indices):
            if len(indices) > 0:
                cluster_probs[idx] = np.sum(prob[indices])
        cluster_probs = cluster_probs / np.sum(cluster_probs)

        return cluster_probs

    def generate_samples_in_clusters(self, *, X_min,
                            cluster_indices, prob, n_to_sample):
        """
        Generate samples in cluster.

        Args:
            X_min (np.array): the minority samples
            cluster_indices (np.array): the cluster indices
            prob (np.array): the probabilities
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        cluster_probs = self.determine_cluster_probabilities(cluster_indices,
                                                                prob)

        clusters_selected = self.random_state.choice(len(cluster_probs),
                                                    n_to_sample,
                                                    p=cluster_probs)
        cluster_unique, cluster_count = np.unique(clusters_selected,
                                                    return_counts=True)

        #n_dim_original = self.n_dim
        samples = []
        for idx, cluster in enumerate(cluster_unique):
            cluster_vectors = X_min[cluster_indices[cluster]]
            within_prob = prob[cluster_indices[cluster]]
            within_prob = within_prob / np.sum(within_prob)
            #self.n_dim = np.min([self.n_dim, cluster_vectors.shape[0]])
            samples.append(self.sample_simplex(X=cluster_vectors,
                            indices=circulant(np.arange(cluster_vectors.shape[0])),
                            n_to_sample=cluster_count[idx],
                            base_weights=within_prob))
            #self.n_dim = n_dim_original

        #samples = []
        #while len(samples) < n_to_sample:
        #    # random sample according to the distribution computed
        #    random_idx = self.random_state.choice(np.arange(len(X_min)),
        #                                          p=prob)
        #
        #    # cluster label of the random minority sample
        #    cluster_label = cluster_labels[random_idx]
        #    if cluster_label == -1:
        #        continue
        #
        #    if len(cluster_indices[cluster_label]) == 0:
        #        continue
        #    elif len(cluster_indices[cluster_label]) == 1:
        #        # if the cluster has only 1 elements, it is repeated
        #        samples.append(X_min[random_idx])
        #        continue
        #
        #    # otherwise a random cluster index is selected for sample
        #    # generation
        #    clus = cluster_indices[cluster_label]
        #    random_neigh_in_clus_idx = self.random_state.choice(clus)
        #    while random_idx == random_neigh_in_clus_idx:
        #        random_neigh_in_clus_idx = self.random_state.choice(clus)
        #
        #    X_rand = X_min[random_idx]
        #    X_in_clus = X_min[random_neigh_in_clus_idx]
        #    samples.append(self.sample_between_points(X_rand, X_in_clus))

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

        _logger.info("%s: Starting TSNE n: %d d: %d",
                        self.__class__.__name__, len(X), len(X[0]))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # do the stochastic embedding
            X_tsne = TSNE(self.n_components, # pylint: disable=invalid-name
                        random_state=self.random_state,
                        perplexity=np.min([10, X.shape[0]-1]),
                        n_iter_without_progress=100,
                        n_iter=500,
                        verbose=0).fit_transform(X)

        X_min = X_tsne[y == self.min_label]
        _logger.info("%s: TSNE finished", self.__class__.__name__)

        try:
            cluster_indices, prob, noise = \
                self.do_clustering(X_tsne, y, X_min)
        except ValueError as valueerror:
            return self.return_copies(X, y, valueerror.args[0])

        # carrying out the sampling
        X_min = X[y == self.min_label]

        samples = self.generate_samples_in_clusters(X_min=X_min,
                                                cluster_indices=cluster_indices,
                                                prob=prob,
                                                n_to_sample=n_to_sample)

        return (np.vstack([np.delete(X, noise, axis=0), np.vstack(samples)]),
                np.hstack([np.delete(y, noise),
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_components': self.n_components,
                'k': self.k,
                'nn_params': self.nn_params,
                'd_cut': self.d_cut,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
