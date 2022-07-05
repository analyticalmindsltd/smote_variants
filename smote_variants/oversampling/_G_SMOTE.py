import numpy as np

from sklearn.metrics import pairwise_distances

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['G_SMOTE']

class G_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{g_smote,
                            author={Sandhan, T. and Choi, J. Y.},
                            booktitle={2014 22nd International Conference on
                                        Pattern Recognition},
                            title={Handling Imbalanced Datasets by Partially
                                    Guided Hybrid Sampling for Pattern
                                    Recognition},
                            year={2014},
                            volume={},
                            number={},
                            pages={1449-1453},
                            keywords={Gaussian processes;learning (artificial
                                        intelligence);pattern classification;
                                        regression analysis;sampling methods;
                                        support vector machines;imbalanced
                                        datasets;partially guided hybrid
                                        sampling;pattern recognition;real-world
                                        domains;skewed datasets;dataset
                                        rebalancing;learning algorithm;
                                        extremely low minority class samples;
                                        classification tasks;extracted hidden
                                        patterns;support vector machine;
                                        logistic regression;nearest neighbor;
                                        Gaussian process classifier;Support
                                        vector machines;Proteins;Pattern
                                        recognition;Kernel;Databases;Gaussian
                                        processes;Vectors;Imbalanced dataset;
                                        protein classification;ensemble
                                        classifier;bootstrapping;Sat-image
                                        classification;medical diagnoses},
                            doi={10.1109/ICPR.2014.258},
                            ISSN={1051-4651},
                            month={Aug}}

    Notes:
        * the non-linear approach is inefficient
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 method='linear',
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
            method (str): 'linear'/'non-linear_2.0' - the float can be any
                                number: standard deviation in the
                                Gaussian-kernel
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        if not method == 'linear' and not method.startswith('non-linear'):
            raise ValueError(self.__class__.__name__ + ": " +
                             'Method parameter %s is not supported' % method)
        elif method.startswith('non-linear'):
            parameter = float(method.split('_')[-1])
            if parameter <= 0:
                message = ("Non-positive non-linear parameter %f is "
                           "not supported") % parameter
                raise ValueError(self.__class__.__name__ + ": " + message)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.method = method
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
                                  'method': ['linear', 'non-linear_0.1',
                                             'non-linear_1.0',
                                             'non-linear_2.0']}
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

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_min)
        ind = nn.kneighbors(X_min, return_distance=False)

        if self.method == 'linear':
            # finding H_l by linear decomposition
            cov = np.cov(X_min, rowvar=False)
            w, v = np.linalg.eig(cov)
            H_l = v[np.argmax(w)]
        else:
            # building a non-linear kernel matrix and finding H_n by its
            # decomposition
            self.sigma = float(self.method.split('_')[-1])
            kernel_matrix = pairwise_distances(X_min)
            kernel_matrix = kernel_matrix/(2.0*self.sigma**2)
            kernel_matrix = np.exp(kernel_matrix)
            try:
                w_k, v_k = np.linalg.eig(kernel_matrix)
            except Exception as e:
                return X.copy(), y.copy()
            H_n = v_k[np.argmax(w_k)]

            def kernel(x, y):
                return np.linalg.norm(x - y)/(2.0*self.sigma**2)

        # generating samples
        samples = []

        def angle(P, n, H_l):
            numerator = np.abs(np.dot(P[n], H_l))
            denominator = np.linalg.norm(P[n])*np.linalg.norm(H_l)
            return np.arccos(numerator/denominator)

        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            # calculating difference vectors from all neighbors
            P = X_min[ind[idx][1:]] - X_min[idx]
            if self.method == 'linear':
                # calculating angles with the principal direction
                thetas = np.array([angle(P, n, H_l) for n in range(len(P))])
            else:
                thetas = []
                # calculating angles of the difference vectors and the
                # principal direction in feature space
                for n in range(len(P)):
                    # calculating representation in feature space
                    feature_vector = np.array(
                        [kernel(X_min[k], P[n]) for k in range(len(X_min))])
                    dp = np.dot(H_n, feature_vector)
                    denom = np.linalg.norm(feature_vector)*np.linalg.norm(H_n)
                    thetas.append(np.arccos(np.abs(dp)/denom))
                thetas = np.array(thetas)

            # using the neighbor with the difference along the most similar
            # direction to the principal direction of the data
            n = np.argmin(thetas)
            X_a = X_min[idx]
            X_b = X_min[ind[idx][1:][n]]
            samples.append(self.sample_between_points_componentwise(X_a, X_b))

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
                'method': self.method,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
