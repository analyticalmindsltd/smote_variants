"""
This module implements the A_SUWO method.
"""

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..base import (NearestNeighborsWithMetricTensor,
                    pairwise_distances_mahalanobis, coalesce)
from ..base import OverSamplingSimplex
from ..base import coalesce_dict

from .._logger import logger
_logger= logger

__all__= ['A_SUWO']

class A_SUWO(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @article{a_suwo,
                        title = "Adaptive semi-unsupervised weighted
                                    oversampling (A-SUWO) for imbalanced
                                    datasets",
                        journal = "Expert Systems with Applications",
                        volume = "46",
                        pages = "405 - 416",
                        year = "2016",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2015.10.031",
                        author = "Iman Nekooeimehr and Susana K. Lai-Yuen",
                        keywords = "Imbalanced dataset, Classification,
                                        Clustering, Oversampling"
                        }

    Notes:
        * Equation (7) misses a division by R_j.
        * It is not specified how to sample from clusters with 1 instances.
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_density_based,
                  OverSamplingSimplex.cat_noise_removal,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 n_clus_maj=7,
                 c_thres=0.8,
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
            ss_params (dict): simplex sampling params
            n_clus_maj (int): number of majority clusters
            c_thres (float): threshold on distances
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                                'within_simplex_sampling': 'random',
                                'gaussian_component': None}

        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params,
                            random_state=random_state,
                            checks=None)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clus_maj, "n_clus_maj", 1)
        self.check_greater_or_equal(c_thres, "c_thres", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.n_clus_maj = n_clus_maj
        self.c_thres = c_thres
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
                                  'n_clus_maj': [5, 7, 9],
                                  'c_thres': [0.5, 0.8]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def remove_noise(self, X, y, nn_params):
        """
        Noise removal

        Args:
            X (np.array): input vectors
            y (np.array): input labels
            nn_params (dict): nearest neighbors parameters

        Returns:
            np.array, np.array: the cleaned X and y
        """
        # fitting nearest neighbors to find neighbors of all samples
        n_neighbors = min([len(X), self.n_neighbors + 1])
        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
        nearestn.fit(X)
        ind = nearestn.kneighbors(X, return_distance=False)

        # identifying as noise those samples which do not have neighbors of
        # the same label
        def noise_func(idx):
            return np.sum(y[ind[idx][1:]] == y[idx]) == 0
        noise = np.where(np.array([noise_func(idx) for idx in range(len(X))]))[0]

        # removing noise
        X = np.delete(X, noise, axis=0)
        y = np.delete(y, noise, axis=0)

        return X, y

    def initialize_clusters(self, X_maj, X_min):
        """
        Initialize the clusters.

        Args:
            X_maj: majority samples
            X_min: minority samples

        Returns:
            list, list: the majority and minority clusters
        """
        n_clus_maj = min([len(X_maj), self.n_clus_maj])

        # clustering majority samples
        aggclus = AgglomerativeClustering(n_clusters=n_clus_maj)
        aggclus.fit(X_maj)
        maj_clusters = [np.where(aggclus.labels_ == i)[0]
                        for i in range(n_clus_maj)]

        # initialize minority clusters
        min_clusters = [np.array([i]) for i in range(len(X_min))]

        return maj_clusters, min_clusters

    def initialize_distances(self,
                                X_maj,
                                X_min,
                                nn_params,
                                *,
                                maj_clusters,
                                min_clusters):
        """
        Initialize the distances

        Args:
            X_maj (np.array): majority samples
            X_min (np.array): minority samples
            nn_params (dict): the nearest neighbors parameters
            maj_clusters (list): the majority clusters
            min_clusters (list): the minority clusters

        Returns:
            np.arary, np.array: the majority and minority distance matrices
        """
        # compute minority distance matrix of cluster
        dm_min = pairwise_distances_mahalanobis(X_min,
                                    tensor=nn_params['metric_tensor'])
        for idx in range(len(dm_min)):
            dm_min[idx, idx] = np.inf

        # compute distance matrix of minority and majority clusters
        dm_maj = np.zeros(shape=(len(X_min), len(maj_clusters)))
        for idx, _ in enumerate(X_min):
            for jdx, maj_cluster in enumerate(maj_clusters):
                pairwd = pairwise_distances_mahalanobis(X_maj[maj_cluster],
                                                        Y=X_min[min_clusters[idx]],
                                                        tensor=nn_params['metric_tensor'])
                dm_maj[idx, jdx] = np.min(pairwd)

        return dm_maj, dm_min

    def determine_threshold(self, X_min, nn_params):
        """
        Determines the threshold.

        Args:
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbors parameters

        Returns:
            float: the threshold
        """
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=len(X_min),
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nearestn.fit(X_min)
        dist, _ = nearestn.kneighbors(X_min)
        d_med = np.median(dist, axis=1)
        return np.mean(d_med)*self.c_thres

    def fix_distribution(self, dist):
        """
        Fixing a distribution.

        Args:
            dist (iterable): the distribution

        Returns:
            np.array: the fixed distribution
        """
        dist = np.nan_to_num(dist, nan=0.0)
        if np.sum(dist) == 0:
            return np.repeat(1.0, len(dist))
        return dist

    def adaptive_sub_cluster_sizing(self, min_clusters, X_min, X_maj):
        """
        Adaptive sub cluster sizing

        Args:
            min_clusters (list): minority clusters
            X_min (np.array): the minority samples
            X_maj (np.array): the majority samples

        Returns:
            np.array: the minority cluster distribution
        """
        eps = []
        # going through all minority clusters
        for clus in min_clusters:
            # checking if cluster size is higher than 1
            if len(clus) > 1:
                kfold = KFold(min([len(clus), 5]),
                                random_state=self.random_state,
                                shuffle=True)
                preds = []
                # executing k-fold cross validation with linear discriminant
                # analysis
                X_c = X_min[clus]
                for train, test in kfold.split(X_c):
                    X_train = np.vstack([X_maj, X_c[train]])
                    y_train = np.hstack([np.repeat(self.maj_label, len(X_maj)),
                                    np.repeat(self.min_label, len(X_c[train]))])
                    lda = LinearDiscriminantAnalysis()
                    lda.fit(X_train, y_train)
                    preds.append(lda.predict(X_c[test]))
                preds = np.hstack(preds)
                # extracting error rate
                eps.append(np.sum(preds == self.maj_label)/len(preds))
            else:
                eps.append(1.0)

        # sampling distribution over clusters
        eps = self.fix_distribution(eps)

        return eps/np.sum(eps)

    def within_cluster(self, X_maj, X_min, nn_params, min_clusters):
        """
        Determine within cluster distributions

        Args:
            X_maj (np.array): the majority samples
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbors parameters
            min_clusters (list): the minority clusters

        Returns:
            list, list: the within cluster distribution and neighbors
        """
        # synthetic instance generation - determining within cluster
        # distribution finding majority neighbor distances of minority
        # samples
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=1,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nearestn.fit(X_maj)
        dist, _ = nearestn.kneighbors(X_min)
        if np.all(dist == 0.0):
            dist[:, :] = 1.0
        else:
            dist[dist == 0.0] = np.min(dist[dist != 0.0])

        dist = dist / X_maj.shape[1]
        dist = 1.0 / dist

        # computing the THs
        THs = [] # pylint: disable=invalid-name
        for clus in min_clusters:
            THs.append(np.mean(dist[clus, 0]))

        # determining within cluster distributions
        within_cluster_dist = []
        for idx, clus in enumerate(min_clusters):
            Gamma = dist[clus, 0] # pylint: disable=invalid-name
            Gamma[Gamma > THs[idx]] = THs[idx]
            within_cluster_dist.append(Gamma / np.sum(Gamma))

        # extracting within cluster neighbors
        within_cluster_neighbors = []
        for clus in min_clusters:
            n_neighbors = min([len(clus), self.n_neighbors])
            nearestn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
            nearestn.fit(X_min[clus])
            within_cluster_neighbors.append(nearestn.kneighbors(X_min[clus])[1])

        return within_cluster_dist, within_cluster_neighbors

    def generate_samples(self,
                        n_to_sample,
                        min_clusters,
                        min_cluster_dist,
                        *,
                        within_dist,
                        within_neigh,
                        X_min):
        """
        Generate samples

        Args:
            n_to_sample (int): the number of samples to generate
            min_clusters (list): minority clusters
            min_cluster_dist (list): minority cluster distribution
            withins (list, list): within cluster distribution and neighbors
            within_cluster_neighbors (list): the within cluster neighbors
            X_min (np.array): the minority samples

        Returns:
            list: the generated samples
        """
        # fixing within class distribution
        within_dist = [self.fix_distribution(w_dist) for w_dist in within_dist]
        within_dist = [w_dist/np.sum(w_dist) for w_dist in within_dist]

        # generate random cluster indices
        cluster_indices = self.random_state.choice(len(min_clusters),
                                                    n_to_sample,
                                                    p=min_cluster_dist)

        cluster_unique, cluster_count = np.unique(cluster_indices,
                                                    return_counts=True)

        samples = []

        for idx, cluster in enumerate(cluster_unique):
            #if len(min_clusters[cluster]) > self.n_dim - 1:
            samples.append(self.sample_simplex(X=X_min[min_clusters[cluster]],
                                                indices=within_neigh[cluster],
                                                n_to_sample=cluster_count[idx],
                                                base_weights=within_dist[cluster]))
            #else:
            #    samp = self.random_state.choice(np.arange(len(within_dist[cluster])),
            #                                    cluster_count[idx],
            #                                    p=within_dist[cluster])
            #    samples.append(X_min[min_clusters[cluster]][samp])

        return np.vstack(samples)

    def cluster_minority_samples(self,
                                min_clusters,
                                dm_maj,
                                dm_min,
                                T # pylint: disable=invalid-name
                                ):
        """
        Clustering of minority samples

        Args:
            min_clusters (list): the minority clusters
            dm_maj (np.array): the majority distance matrix
            dm_min (np.array): the minority distance matrix
            T (float): the threshold

        Returns:
            list: the minority clusters
        """
        # do the clustering of minority samples
        while True:
            # finding minimum distance between minority clusters
            pi = np.min(dm_min) # pylint: disable=invalid-name

            # if the minimum distance is higher than the threshold, stop
            if pi > T:
                break

            # find cluster pair of minimum distance
            min_dist_pair = np.where(dm_min == pi)
            min_i = min_dist_pair[0][0]
            min_j = min_dist_pair[1][0]

            # Step 3 - find majority clusters closer than pi
            A = np.where(np.logical_and(dm_maj[min_i] < pi, # pylint: disable=invalid-name
                                        dm_maj[min_j] < pi))[0]

            # Step 4 - checking if there is a majority cluster between the
            # minority ones
            if len(A) > 0:
                dm_min[min_i, min_j] = np.inf
                dm_min[min_j, min_i] = np.inf
            else:
                # Step 5
                # unifying minority clusters
                min_clusters[min_i] = np.hstack([min_clusters[min_i],
                                                 min_clusters[min_j]])
                # removing one of them
                #min_clusters = np.delete(min_clusters, [min_j], axis=0)
                del min_clusters[min_j]

                # updating the minority distance matrix
                dm_min[min_i] = np.min(np.vstack([dm_min[min_i],
                                                  dm_min[min_j]]), axis=0)
                dm_min[:, min_i] = dm_min[min_i]
                # removing jth row and column (merged in i)
                dm_min = np.delete(dm_min, min_j, axis=0)
                dm_min = np.delete(dm_min, min_j, axis=1)

                # fixing the diagonal elements
                for idx in range(len(dm_min)):
                    dm_min[idx, idx] = np.inf

                # updating the minority-majority distance matrix
                dm_maj[min_i] = np.min(np.vstack([dm_maj[min_i],
                                                  dm_maj[min_j]]), axis=0)
                dm_maj = np.delete(dm_maj, min_j, axis=0)

        return min_clusters

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
            return self.return_copies(X, y, "no need for sampling")

        X_orig, y_orig = X, y

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        X, y = self.remove_noise(X, y, nn_params)

        # extracting modified minority and majority datasets
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        if len(X_min) == 0:
            _logger.info("%s: All minority samples removed as noise",
                            self.__class__.__name__)
            return X_orig.copy(), y_orig.copy()

        maj_clusters, min_clusters = self.initialize_clusters(X_maj, X_min)

        # I think this can never happen
        # if len(maj_clusters) == 0:
        #    _logger.info("%s: Number of majority clusters is null",
        #                self.__class__.__name__)
        #    return X_orig.copy(), y_orig.copy()

        dms = self.initialize_distances(X_maj,
                                        X_min,
                                        nn_params,
                                        maj_clusters=maj_clusters,
                                        min_clusters=min_clusters)

        min_clusters = self.cluster_minority_samples(min_clusters,
                                                *dms,
                                                self.determine_threshold(X_min,
                                                                    nn_params))

        # adaptive sub-cluster sizing
        min_cluster_dist = self.adaptive_sub_cluster_sizing(min_clusters,
                                                            X_min,
                                                            X_maj)

        within_dist, within_neigh = self.within_cluster(X_maj,
                                                        X_min,
                                                        nn_params,
                                                        min_clusters)

        # do the sampling
        return (np.vstack([X, np.vstack(self.generate_samples(n_to_sample,
                                                    min_clusters,
                                                    min_cluster_dist,
                                                    within_dist=within_dist,
                                                    within_neigh=within_neigh,
                                                    X_min=X_min))]),
                np.hstack([y, np.repeat(self.min_label, n_to_sample)]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_clus_maj': self.n_clus_maj,
                'c_thres': self.c_thres,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
