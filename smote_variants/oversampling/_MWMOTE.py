import numpy as np

from sklearn.cluster import KMeans

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['MWMOTE']

class MWMOTE(OverSampling):
    """
    References:
        * BibTex::

            @ARTICLE{mwmote,
                        author={Barua, S. and Islam, M. M. and Yao, X. and
                                Murase, K.},
                        journal={IEEE Transactions on Knowledge and Data
                                Engineering},
                        title={MWMOTE--Majority Weighted Minority Oversampling
                                Technique for Imbalanced Data Set Learning},
                        year={2014},
                        volume={26},
                        number={2},
                        pages={405-425},
                        keywords={learning (artificial intelligence);pattern
                                    clustering;sampling methods;AUC;area under
                                    curve;ROC;receiver operating curve;G-mean;
                                    geometric mean;minority class cluster;
                                    clustering approach;weighted informative
                                    minority class samples;Euclidean distance;
                                    hard-to-learn informative minority class
                                    samples;majority class;synthetic minority
                                    class samples;synthetic oversampling
                                    methods;imbalanced learning problems;
                                    imbalanced data set learning;
                                    MWMOTE-majority weighted minority
                                    oversampling technique;Sampling methods;
                                    Noise measurement;Boosting;Simulation;
                                    Complexity theory;Interpolation;Abstracts;
                                    Imbalanced learning;undersampling;
                                    oversampling;synthetic sample generation;
                                    clustering},
                        doi={10.1109/TKDE.2012.232},
                        ISSN={1041-4347},
                        month={Feb}}

    Notes:
        * The original method was not prepared for the case of having clusters
            of 1 elements.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 k1=5,
                 k2=5,
                 k3=5,
                 M=10,
                 cf_th=5.0,
                 cmax=10.0,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            k1 (int): parameter of the NearestNeighbors component
            k2 (int): parameter of the NearestNeighbors component
            k3 (int): parameter of the NearestNeighbors component
            M (int): number of clusters
            cf_th (float): cutoff threshold
            cmax (float): maximum closeness value
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(k1, 'k1', 1)
        self.check_greater_or_equal(k2, 'k2', 1)
        self.check_greater_or_equal(k3, 'k3', 1)
        self.check_greater_or_equal(M, 'M', 1)
        self.check_greater_or_equal(cf_th, 'cf_th', 0)
        self.check_greater_or_equal(cmax, 'cmax', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.M = M
        self.cf_th = cf_th
        self.cmax = cmax
        self.nn_params = nn_params
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
                                  'k1': [5, 9],
                                  'k2': [5, 9],
                                  'k3': [5, 9],
                                  'M': [4, 10],
                                  'cf_th': [5.0],
                                  'cmax': [10.0]}
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
        X_maj = X[y == self.maj_label]

        minority = np.where(y == self.min_label)[0]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # Step 1
        n_neighbors = min([len(X), self.k1 + 1])
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        ind1 = nn.kneighbors(X, return_distance=False)

        # Step 2
        arr = [i for i in minority if np.sum(y[ind1[i][1:]] == self.min_label)]
        filtered_minority = np.array(arr)

        if len(filtered_minority) == 0:
            _logger.info(self.__class__.__name__ + ": " +
                         "filtered_minority array is empty")
            return X.copy(), y.copy()

        # Step 3 - ind2 needs to be indexed by indices of the lengh of X_maj
        nn_maj= NearestNeighborsWithMetricTensor(n_neighbors=self.k2, 
                                                    n_jobs=self.n_jobs, 
                                                    **(nn_params))
        nn_maj.fit(X_maj)
        ind2 = nn_maj.kneighbors(X[filtered_minority], return_distance=False)

        # Step 4
        border_majority = np.unique(ind2.flatten())

        # Step 5 - ind3 needs to be indexed by indices of the length of X_min
        n_neighbors = min([self.k3, len(X_min)])
        nn_min = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                            n_jobs=self.n_jobs, 
                                                            **(nn_params))
        nn_min.fit(X_min)
        ind3 = nn_min.kneighbors(X_maj[border_majority], return_distance=False)

        # Step 6 - informative minority indexes X_min
        informative_minority = np.unique(ind3.flatten())

        def closeness_factor(y, x, cf_th=self.cf_th, cmax=self.cmax):
            """
            Closeness factor according to the Eq (6)

            Args:
                y (np.array): training instance (border_majority)
                x (np.array): training instance (informative_minority)
                cf_th (float): cutoff threshold
                cmax (float): maximum values

            Returns:
                float: closeness factor
            """
            d = np.linalg.norm(y - x)/len(y)
            if d == 0.0:
                d = 0.1
            if 1.0/d < cf_th:
                f = 1.0/d
            else:
                f = cf_th
            return f/cf_th*cmax

        # Steps 7 - 9
        _logger.info(self.__class__.__name__ + ": " +
                     'computing closeness factors')
        closeness_factors = np.zeros(
            shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            bm_i = border_majority[i]
            for j in range(len(informative_minority)):
                im_j = informative_minority[j]
                closeness_factors[i, j] = closeness_factor(X_maj[bm_i],
                                                           X_min[im_j])

        _logger.info(self.__class__.__name__ + ": " +
                     'computing information weights')
        information_weights = np.zeros(
            shape=(len(border_majority), len(informative_minority)))
        for i in range(len(border_majority)):
            norm_factor = np.sum(closeness_factors[i, :])
            for j in range(len(informative_minority)):
                cf_ij = closeness_factors[i, j]
                information_weights[i, j] = cf_ij**2/norm_factor

        selection_weights = np.sum(information_weights, axis=0)
        selection_probabilities = selection_weights/np.sum(selection_weights)

        # Step 10
        _logger.info(self.__class__.__name__ + ": " + 'do clustering')
        n_clusters = min([len(X_min), self.M])
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self._random_state_init)
        kmeans.fit(X_min)
        imin_labels = kmeans.labels_[informative_minority]

        clusters = [np.where(imin_labels == i)[0]
                    for i in range(np.max(kmeans.labels_)+1)]

        # Step 11
        samples = []

        # Step 12
        for i in range(n_to_sample):
            random_index = self.random_state.choice(informative_minority,
                                                    p=selection_probabilities)
            cluster_label = kmeans.labels_[random_index]
            cluster = clusters[cluster_label]
            random_index_in_cluster = self.random_state.choice(cluster)
            X_random = X_min[random_index]
            X_random_cluster = X_min[random_index_in_cluster]
            samples.append(self.sample_between_points(X_random,
                                                      X_random_cluster))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k1': self.k1,
                'k2': self.k2,
                'k3': self.k3,
                'M': self.M,
                'cf_th': self.cf_th,
                'cmax': self.cmax,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
