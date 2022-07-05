import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling

from .._logger import logger
_logger= logger

__all__= ['Assembled_SMOTE']

class Assembled_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{assembled_smote,
                            author={Zhou, B. and Yang, C. and Guo, H. and
                                        Hu, J.},
                            booktitle={The 2013 International Joint Conference
                                        on Neural Networks (IJCNN)},
                            title={A quasi-linear SVM combined with assembled
                                    SMOTE for imbalanced data classification},
                            year={2013},
                            volume={},
                            number={},
                            pages={1-7},
                            keywords={approximation theory;interpolation;
                                        pattern classification;sampling
                                        methods;support vector machines;trees
                                        (mathematics);quasilinear SVM;
                                        assembled SMOTE;imbalanced dataset
                                        classification problem;oversampling
                                        method;quasilinear kernel function;
                                        approximate nonlinear separation
                                        boundary;mulitlocal linear boundaries;
                                        interpolation;data distribution
                                        information;minimal spanning tree;
                                        local linear partitioning method;
                                        linear separation boundary;synthetic
                                        minority class samples;oversampled
                                        dataset classification;standard SVM;
                                        composite quasilinear kernel function;
                                        artificial data datasets;benchmark
                                        datasets;classification performance
                                        improvement;synthetic minority
                                        over-sampling technique;Support vector
                                        machines;Kernel;Merging;Standards;
                                        Sociology;Statistics;Interpolation},
                            doi={10.1109/IJCNN.2013.6707035},
                            ISSN={2161-4407},
                            month={Aug}}

    Notes:
        * Absolute value of the angles extracted should be taken.
            (implemented this way)
        * It is not specified how many samples are generated in the various
            clusters.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_borderline,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 pop=2,
                 thres=0.3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            pop (int): lower threshold on cluster sizes
            thres (float): threshold on angles
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(pop, "pop", 1)
        self.check_in_range(thres, "thres", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.pop = pop
        self.thres = thres
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
                                  'pop': [2, 4, 5],
                                  'thres': [0.1, 0.3, 0.5]}
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

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors+1])
        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        ind = nn.kneighbors(X_min, return_distance=False)

        # finding the set of border and non-border minority elements
        n_min_neighbors = [np.sum(y[ind[i]] == self.min_label)
                           for i in range(len(ind))]
        border_mask = np.logical_not(np.array(n_min_neighbors) == n_neighbors)
        X_border = X_min[border_mask]
        X_non_border = X_min[np.logical_not(border_mask)]

        if len(X_border) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "X_border is empty")
            return X.copy(), y.copy()

        # initializing clustering
        clusters = [np.array([i]) for i in range(len(X_border))]
        dm = pairwise_distances_mahalanobis(X_border, 
                                            tensor=nn_params.get('metric_tensor', None))
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # do the clustering
        while len(dm) > 1 and np.min(dm) < np.inf:
            # extracting coordinates of clusters with the minimum distance
            min_coord = np.where(dm == np.min(dm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # checking the size of clusters to see if they should be merged
            if (len(clusters[merge_a]) < self.pop
                    or len(clusters[merge_b]) < self.pop):
                # if both clusters are small, do the merge
                clusters[merge_a] = np.hstack([clusters[merge_a],
                                               clusters[merge_b]])
                del clusters[merge_b]
                # update the distance matrix accordingly
                dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]),
                                     axis=0)
                dm[:, merge_a] = dm[merge_a]
                # remove columns
                dm = np.delete(dm, merge_b, axis=0)
                dm = np.delete(dm, merge_b, axis=1)
                # fix the diagonal entries
                for i in range(len(dm)):
                    dm[i, i] = np.inf
            else:
                # otherwise find principal directions
                pca_a = PCA(n_components=1).fit(X_border[clusters[merge_a]])
                pca_b = PCA(n_components=1).fit(X_border[clusters[merge_b]])
                # extract the angle of principal directions
                numerator = np.dot(pca_a.components_[0], pca_b.components_[0])
                denominator = np.linalg.norm(pca_a.components_[0])
                denominator *= np.linalg.norm(pca_b.components_[0])
                angle = abs(numerator/denominator)
                # check if angle if angle is above a specific threshold
                if angle > self.thres:
                    # do the merge
                    clusters[merge_a] = np.hstack([clusters[merge_a],
                                                   clusters[merge_b]])
                    del clusters[merge_b]
                    # update the distance matrix acoordingly
                    dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]),
                                         axis=0)
                    dm[:, merge_a] = dm[merge_a]
                    # remove columns
                    dm = np.delete(dm, merge_b, axis=0)
                    dm = np.delete(dm, merge_b, axis=1)
                    # fixing the digaonal entries
                    for i in range(len(dm)):
                        dm[i, i] = np.inf
                else:
                    # changing the distance of clusters to fininte
                    dm[merge_a, merge_b] = np.inf
                    dm[merge_b, merge_a] = np.inf

        # extract vectors belonging to the various clusters
        vectors = [X_border[c] for c in clusters if len(c) > 0]
        # adding non-border samples
        if len(X_non_border) > 0:
            vectors.append(X_non_border)

        # extract cluster sizes and calculating point distribution in clusters
        # the last element of the clusters is the set of non-border xamples
        cluster_sizes = np.array([len(v) for v in vectors])
        densities = cluster_sizes/np.sum(cluster_sizes)

        # extracting nearest neighbors in clusters
        def fit_knn(vectors):
            n_neighbors = min([self.n_neighbors + 1, len(vectors)])
            nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
            return nn.fit(vectors).kneighbors(vectors)

        nns = [fit_knn(v) for v in vectors]

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            cluster_idx = self.random_state.choice(len(vectors), p=densities)
            len_cluster = len(vectors[cluster_idx])
            sample_idx = self.random_state.choice(np.arange(len_cluster))

            if len_cluster > 1:
                choose_from = nns[cluster_idx][1][sample_idx][1:]
                random_neighbor_idx = self.random_state.choice(choose_from)
            else:
                random_neighbor_idx = sample_idx

            X_a = vectors[cluster_idx][sample_idx]
            X_b = vectors[cluster_idx][random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

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
                'pop': self.pop,
                'thres': self.thres,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
