import numpy as np

from sklearn.cluster import KMeans

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling

from .._logger import logger
_logger= logger

__all__= ['LVQ_SMOTE']

class LVQ_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{lvq_smote,
                              title={LVQ-SMOTE – Learning Vector Quantization
                                    based Synthetic Minority Over–sampling
                                    Technique for biomedical data},
                              author={Munehiro Nakamura and Yusuke Kajiwara
                                     and Atsushi Otsuka and Haruhiko Kimura},
                              booktitle={BioData Mining},
                              year={2013}
                            }

    Notes:
        * This implementation is only a rough approximation of the method
            described in the paper. The main problem is that the paper uses
            many datasets to find similar patterns in the codebooks and
            replicate patterns appearing in other datasets to the imbalanced
            datasets based on their relative position compared to the codebook
            elements. What we do is clustering the minority class to extract
            a codebook as kmeans cluster means, then, find pairs of codebook
            elements which have the most similar relative position to a
            randomly selected pair of codebook elements, and translate nearby
            minority samples from the neighborhood one pair of codebook
            elements to the neighborood of another pair of codebook elements.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_application]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_clusters=10,
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
            n_clusters (int): number of clusters in vector quantization
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_clusters, "n_clusters", 3)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_clusters = n_clusters
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
                                  'n_clusters': [4, 8, 12]}
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

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # clustering X_min to extract codebook
        n_clusters = min([len(X_min), self.n_clusters])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self._random_state_init)
        kmeans.fit(X_min)
        codebook = kmeans.cluster_centers_

        # get nearest neighbors of minority samples to codebook samples
        n_neighbors = min([len(X_min), self.n_neighbors])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        indices = nn.kneighbors(codebook, return_distance=False)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # randomly selecting a pair of codebook elements
            cb_0, cb_1 = self.random_state.choice(
                list(range(len(codebook))), 2, replace=False)
            diff = codebook[cb_0] - codebook[cb_1]
            min_dist = np.inf
            min_0 = None
            # finding another pair of codebook elements with similar offset
            for i in range(len(codebook)):
                for j in range(len(codebook)):
                    if cb_0 != i and cb_0 != j and cb_1 != i and cb_1 != j:
                        dd = np.linalg.norm(diff - (codebook[i] - codebook[j]))
                        if dd < min_dist:
                            min_dist = dd
                            min_0 = self.random_state.choice([i, j])

            # translating a random neighbor of codebook element min_0 to
            # the neighborhood of point_0
            random_index = self.random_state.randint(len(indices[min_0]))
            sample = X_min[indices[min_0][random_index]]
            point_0 = codebook[cb_0] + (sample - codebook[min_0])

            samples.append(point_0)

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
                'random_state': self._random_state_init}
