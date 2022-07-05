import numpy as np

from sklearn.manifold import TSNE
import scipy.signal as ssignal

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['MOT2LD']

class MOT2LD(OverSampling):
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

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_components=2,
                 k=5,
                 nn_params={},
                 d_cut='auto',
                 n_jobs=1,
                 random_state=None):
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
            d_cut (float/str): distance cut value/'auto' for automated
                                selection
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_components, 'n_component', 1)
        self.check_greater_or_equal(k, 'k', 1)
        if isinstance(d_cut, float) or isinstance(d_cut, int):
            if d_cut <= 0:
                raise ValueError(self.__class__.__name__ +
                                 ": " + 'Non-positive d_cut is not allowed')
        elif d_cut != 'auto':
            raise ValueError(self.__class__.__name__ + ": " +
                             'd_cut value %s not implemented' % d_cut)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.k = k
        self.nn_params = nn_params
        self.d_cut = d_cut
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
                                  'n_components': [2],
                                  'k': [3, 5, 7],
                                  'd_cut': ['auto']}
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

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        _logger.info(self.__class__.__name__ + ": " +
                     ("starting TSNE n: %d d: %d" % (len(X), len(X[0]))))
        # do the stochastic embedding
        X_tsne = TSNE(self.n_components,
                      random_state=self.random_state,
                      perplexity=10,
                      n_iter_without_progress=100,
                      n_iter=500,
                      verbose=0).fit_transform(X)
        
        X_min = X_tsne[y == self.min_label]
        _logger.info(self.__class__.__name__ + ": " + "TSNE finished")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X_tsne, y)

        # fitting nearest neighbors model for all training data
        n_neighbors = min([len(X_min), self.k + 1])
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_tsne)
        distances, indices = nn.kneighbors(X_min)

        if isinstance(self.d_cut, float):
            d_cut = self.d_cut
        elif self.d_cut == 'auto':
            d_cut = np.max(distances[:, 1])

        # fitting nearest neighbors model to the minority data
        nn_min= NearestNeighborsWithMetricTensor(n_neighbors=len(X_min), 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
        nn_min.fit(X_min)
        distances_min, indices_min = nn_min.kneighbors(X_min)

        def n_rad_neighbors(x):
            x = x.reshape(1, -1)
            return len(nn.radius_neighbors(x, d_cut, return_distance=False)[0])

        # extracting the number of neighbors in a given radius
        rho = np.array([n_rad_neighbors(x) for x in X_min])
        closest_highest = []
        delta = []

        # implementation of the density peak clustering algorithm
        # based on http://science.sciencemag.org/content/344/6191/1492.full
        for i in range(len(rho)):
            closest_neighbors = indices_min[i]
            closest_densities = rho[closest_neighbors]
            closest_highs = np.where(closest_densities > rho[i])[0]

            if len(closest_highs) > 0:
                closest_highest.append(closest_highs[0])
                delta.append(distances_min[i][closest_highs[0]])
            else:
                closest_highest.append(-1)
                delta.append(np.max(distances_min))

        to_sort = zip(rho, delta, np.arange(len(rho)))
        r, d, idx = zip(*sorted(to_sort, key=lambda x: x[0]))
        r, d, idx = np.array(r), np.array(d), np.array(idx)

        if len(d) < 3:
            return X.copy(), y.copy()

        widths = np.arange(1, int(len(r)/2))
        peak_indices = np.array(ssignal.find_peaks_cwt(d, widths=widths))

        if len(peak_indices) == 0:
            _logger.info(self.__class__.__name__ + ": " + "no peaks found")
            return X.copy(), y.copy()

        cluster_center_indices = idx[peak_indices]
        cluster_centers = X_min[cluster_center_indices]

        # finding closest cluster center to minority points and deriving
        # cluster labels
        nn_cluster= NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params))
        nn_cluster.fit(cluster_centers)
        ind_cluster = nn_cluster.kneighbors(X_min, return_distance=False)
        cluster_labels = ind_cluster[:, 0]

        # computing local minority counts and determining noisy samples
        def n_min_y(i):
            return np.sum(y[indices[i][1:]] == self.min_label)

        local_minority_count = np.array(
            [n_min_y(i) for i in range(len(X_min))])

        noise = np.where(np.logical_or(rho == 1, local_minority_count == 0))[0]

        # determining importance scores
        importance = local_minority_count/rho
        prob = importance
        prob[noise] = 0.0
        prob = prob/np.sum(prob)

        # extracting cluster indices
        cluster_indices = [np.where(cluster_labels == i)[0]
                           for i in range(np.max(cluster_labels) + 1)]
        # removing noise from clusters
        cluster_indices = [list(set(c).difference(set(noise)))
                           for c in cluster_indices]

        # checking if clustering is empty
        empty_clustering = True
        for i in range(len(cluster_indices)):
            if len(cluster_indices[i]) > 0:
                empty_clustering = False

        if empty_clustering:
            _logger.info(self.__class__.__name__ + ": " + "Empty clustering")
            return X.copy(), y.copy()

        cluster_sizes = np.array([len(c) for c in cluster_indices])
        cluster_indices_size_0 = np.where(cluster_sizes == 0)[0]
        for i in range(len(prob)):
            if cluster_labels[i] in cluster_indices_size_0:
                prob[i] = 0.0
        prob = prob/np.sum(prob)

        # carrying out the sampling
        X_min = X[y == self.min_label]
        samples = []
        while len(samples) < n_to_sample:
            # random sample according to the distribution computed
            random_idx = self.random_state.choice(np.arange(len(X_min)),
                                                  p=prob)

            # cluster label of the random minority sample
            cluster_label = cluster_labels[random_idx]
            if cluster_label == -1:
                continue

            if len(cluster_indices[cluster_label]) == 0:
                continue
            elif len(cluster_indices[cluster_label]) == 1:
                # if the cluster has only 1 elements, it is repeated
                samples.append(X_min[random_idx])
                continue
            else:
                # otherwise a random cluster index is selected for sample
                # generation
                clus = cluster_indices[cluster_label]
                random_neigh_in_clus_idx = self.random_state.choice(clus)
                while random_idx == random_neigh_in_clus_idx:
                    random_neigh_in_clus_idx = self.random_state.choice(clus)

                X_rand = X_min[random_idx]
                X_in_clus = X_min[random_neigh_in_clus_idx]
                samples.append(self.sample_between_points(X_rand, X_in_clus))

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
                'random_state': self._random_state_init}
