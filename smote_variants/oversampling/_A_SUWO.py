import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['A_SUWO']

class A_SUWO(OverSampling):
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

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_clus_maj=7,
                 c_thres=0.8,
                 n_jobs=1,
                 random_state=None):
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
            n_clus_maj (int): number of majority clusters
            c_thres (float): threshold on distances
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clus_maj, "n_clus_maj", 1)
        self.check_greater_or_equal(c_thres, "c_thres", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_clus_maj = n_clus_maj
        self.c_thres = c_thres
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
                                  'n_clus_maj': [5, 7, 9],
                                  'c_thres': [0.5, 0.8]}
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

        X_orig, y_orig = X, y

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors to find neighbors of all samples
        n_neighbors = min([len(X), self.n_neighbors + 1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        dist, ind = nn.kneighbors(X)

        # identifying as noise those samples which do not have neighbors of
        # the same label
        def noise_func(i):
            return np.sum(y[ind[i][1:]] == y[i]) == 0
        noise = np.where(np.array([noise_func(i) for i in range(len(X))]))[0]

        # removing noise
        X = np.delete(X, noise, axis=0)
        y = np.delete(y, noise, axis=0)

        # extracting modified minority and majority datasets
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        if len(X_min) == 0:
            _logger.info("All minority samples removed as noise")
            return X_orig.copy(), y_orig.copy()

        n_clus_maj = min([len(X_maj), self.n_clus_maj])

        # clustering majority samples
        ac = AgglomerativeClustering(n_clusters=n_clus_maj)
        ac.fit(X_maj)
        maj_clusters = [np.where(ac.labels_ == i)[0]
                        for i in range(n_clus_maj)]

        if len(maj_clusters) == 0:
            return X_orig.copy(), y_orig.copy()

        # initialize minority clusters
        min_clusters = [np.array([i]) for i in range(len(X_min))]

        # compute minority distance matrix of cluster
        dm_min = pairwise_distances_mahalanobis(X_min, tensor=nn_params['metric_tensor'])
        for i in range(len(dm_min)):
            dm_min[i, i] = np.inf

        # compute distance matrix of minority and majority clusters
        dm_maj = np.zeros(shape=(len(X_min), len(maj_clusters)))
        for i in range(len(X_min)):
            for j in range(len(maj_clusters)):
                pairwd = pairwise_distances_mahalanobis(X_min[min_clusters[i]],
                                                        X_maj[maj_clusters[j]],
                                                        tensor=nn_params['metric_tensor'])
                dm_maj[i, j] = np.min(pairwd)

        # compute threshold
        nn = NearestNeighborsWithMetricTensor(n_neighbors=len(X_min), 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)
        d_med = np.median(dist, axis=1)
        T = np.mean(d_med)*self.c_thres

        # do the clustering of minority samples
        while True:
            # finding minimum distance between minority clusters
            pi = np.min(dm_min)

            # if the minimum distance is higher than the threshold, stop
            if pi > T:
                break

            # find cluster pair of minimum distance
            min_dist_pair = np.where(dm_min == pi)
            min_i = min_dist_pair[0][0]
            min_j = min_dist_pair[1][0]

            # Step 3 - find majority clusters closer than pi
            A = np.where(np.logical_and(dm_maj[min_i] < pi,
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
                for i in range(len(dm_min)):
                    dm_min[i, i] = np.inf

                # updating the minority-majority distance matrix
                dm_maj[min_i] = np.min(np.vstack([dm_maj[min_i],
                                                  dm_maj[min_j]]), axis=0)
                dm_maj = np.delete(dm_maj, min_j, axis=0)

        # adaptive sub-cluster sizing
        eps = []
        # going through all minority clusters
        for c in min_clusters:
            # checking if cluster size is higher than 1
            if len(c) > 1:
                k = min([len(c), 5])
                kfold = KFold(k, random_state=self.random_state, shuffle=True)
                preds = []
                # executing k-fold cross validation with linear discriminant
                # analysis
                X_c = X_min[c]
                for train, test in kfold.split(X_c):
                    X_train = np.vstack([X_maj, X_c[train]])
                    y_train_maj = np.repeat(self.maj_label, len(X_maj))
                    y_train_min = np.repeat(self.min_label, len(X_c[train]))
                    y_train = np.hstack([y_train_maj, y_train_min])
                    ld = LinearDiscriminantAnalysis()
                    ld.fit(X_train, y_train)
                    preds.append(ld.predict(X_c[test]))
                preds = np.hstack(preds)
                # extracting error rate
                eps.append(np.sum(preds == self.maj_label)/len(preds))
            else:
                eps.append(1.0)

        # sampling distribution over clusters
        if np.sum(eps) == 0.0:
            eps= np.repeat(1.0, len(eps))
        min_cluster_dist = eps/np.sum(eps)

        # synthetic instance generation - determining within cluster
        # distribution finding majority neighbor distances of minority
        # samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_maj)
        dist, ind = nn.kneighbors(X_min)
        dist = dist/len(X[0])
        dist = 1.0/dist

        # computing the THs
        THs = []
        for c in min_clusters:
            THs.append(np.mean(dist[c, 0]))

        # determining within cluster distributions
        within_cluster_dist = []
        for i, c in enumerate(min_clusters):
            Gamma = dist[c, 0]
            Gamma[Gamma > THs[i]] = THs[i]
            within_cluster_dist.append(Gamma/np.sum(Gamma))

        # extracting within cluster neighbors
        within_cluster_neighbors = []
        for c in min_clusters:
            n_neighbors = min([len(c), self.n_neighbors])
            nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
            nn.fit(X_min[c])
            within_cluster_neighbors.append(nn.kneighbors(X_min[c])[1])

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # choose random cluster index
            cluster_idx = self.random_state.choice(
                np.arange(len(min_clusters)), p=min_cluster_dist)
            if len(min_clusters[cluster_idx]) > 1:
                # if the cluster has at least two elements
                domain = np.arange(len(min_clusters[cluster_idx]))
                distribution = within_cluster_dist[cluster_idx]
                if np.any(np.isnan(distribution)):
                    distribution= np.nan_to_num(distribution, nan=0.0)
                if np.sum(distribution) == 0.0:
                    distribution= np.repeat(1.0, len(distribution))
                    distribution= distribution/len(distribution)
                sample_idx = self.random_state.choice(domain, p=distribution)

                domain = within_cluster_neighbors[cluster_idx][sample_idx][1:]
                neighbor_idx = self.random_state.choice(domain)
                point = X_min[min_clusters[cluster_idx][sample_idx]]
                neighbor = X_min[min_clusters[cluster_idx][neighbor_idx]]
                samples.append(self.sample_between_points(point, neighbor))
            else:
                samples.append(X_min[min_clusters[cluster_idx][0]])

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
                'n_clus_maj': self.n_clus_maj,
                'c_thres': self.c_thres,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
