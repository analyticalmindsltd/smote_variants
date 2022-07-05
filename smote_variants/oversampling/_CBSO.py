import numpy as np

from sklearn.metrics import pairwise_distances

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['CBSO']

class CBSO(OverSampling):
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

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 C_p=1.3,
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
            C_p (float): used to set the threshold of clustering
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater(C_p, "C_p", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.C_p = C_p
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
                                  'C_p': [0.8, 1.0, 1.3, 1.6]}
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

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model to find neighbors of minority points
        nn = NearestNeighborsWithMetricTensor(n_neighbors=self.n_neighbors + 1, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        ind = nn.kneighbors(X_min, return_distance=False)

        # extracting the number of majority neighbors
        weights = [np.sum(y[ind[i][1:]] == self.maj_label)
                   for i in range(len(X_min))]
        
        weights= np.nan_to_num(weights, nan=0.0)
        
        if np.sum(weights) == 0:
            weights= np.repeat(1, len(weights))
        
        # determine distribution of generating data
        weights = weights/np.sum(weights)
        
        # do the clustering
        nn = NearestNeighborsWithMetricTensor(n_neighbors=2, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        d_avg = np.mean(nn.kneighbors(X_min)[0][:, 1])
        T_h = d_avg*self.C_p

        # initiating clustering
        clusters = [np.array([i]) for i in range(len(X_min))]
        dm = pairwise_distances_mahalanobis(X_min, 
                                            tensor=nn_params.get('metric_tensor', None))

        # setting the diagonal of the distance matrix to infinity
        for i in range(len(dm)):
            dm[i, i] = np.inf

        # starting the clustering iteration
        while True:
            # finding the cluster pair with the smallest distance
            min_coord = np.where(dm == np.min(dm))
            merge_a = min_coord[0][0]
            merge_b = min_coord[1][0]

            # check termination conditions
            if dm[merge_a, merge_b] > T_h or len(dm) == 1:
                break

            # merging the clusters
            clusters[merge_a] = np.hstack(
                [clusters[merge_a], clusters[merge_b]])
            # removing one of them
            del clusters[merge_b]
            # adjusting the distances in the distance matrix
            dm[merge_a] = np.min(np.vstack([dm[merge_a], dm[merge_b]]), axis=0)
            dm[:, merge_a] = dm[merge_a]
            # removing the row and column corresponding to one of the
            # merged clusters
            dm = np.delete(dm, merge_b, axis=0)
            dm = np.delete(dm, merge_b, axis=1)
            # updating the diagonal
            for i in range(len(dm)):
                dm[i, i] = np.inf

        # extracting cluster labels
        labels = np.zeros(len(X_min)).astype(int)
        for i in range(len(clusters)):
            for j in clusters[i]:
                labels[j] = i

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)), p=weights)
            if len(clusters[labels[idx]]) <= 1:
                samples.append(X_min[idx])
                continue
            else:
                random_idx = self.random_state.choice(clusters[labels[idx]])
                while random_idx == idx:
                    random_idx = self.random_state.choice(
                        clusters[labels[idx]])
            samples.append(self.sample_between_points(
                X_min[idx], X_min[random_idx]))

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
                'C_p': self.C_p,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
