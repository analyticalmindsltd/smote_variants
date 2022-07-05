import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['AND_SMOTE']

class AND_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{and_smote,
                             author = {Yun, Jaesub and Ha,
                                 Jihyun and Lee, Jong-Seok},
                             title = {Automatic Determination of Neighborhood
                                        Size in SMOTE},
                             booktitle = {Proceedings of the 10th International
                                            Conference on Ubiquitous
                                            Information Management and
                                            Communication},
                             series = {IMCOM '16},
                             year = {2016},
                             isbn = {978-1-4503-4142-4},
                             location = {Danang, Viet Nam},
                             pages = {100:1--100:8},
                             articleno = {100},
                             numpages = {8},
                             doi = {10.1145/2857546.2857648},
                             acmid = {2857648},
                             publisher = {ACM},
                             address = {New York, NY, USA},
                             keywords = {SMOTE, imbalanced learning, synthetic
                                            data generation},
                            }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self, 
                 proportion=1.0, 
                 *,
                 K=15, 
                 nn_params={},
                 n_jobs=1, 
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            K (int): maximum number of nearest neighbors
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
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(K, "K", 2)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.K = K
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
                                  'K': [9, 15, 21]}
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

        K = min([len(X_min), self.K])
        # find K nearest neighbors of all samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=K, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        ind = nn.kneighbors(X, return_distance=False)

        min_ind = np.where(y == self.min_label)[0]

        # Executing the algorithm
        kappa = []
        for i in range(len(min_ind)):
            regions_min = []
            regions_maj = []

            for j in range(1, K):
                # continueing if the label of the neighbors is minority
                if y[ind[min_ind[i]][j]] != self.min_label:
                    continue

                # region coordinates
                reg = np.hstack([min_ind[i], ind[min_ind[i]][j]])
                # compute corner points
                reg_min = np.min(X[reg])
                reg_max = np.max(X[reg])

                r_min = []
                r_maj = []
                # all the points in the region must be among the neighbors
                # what we do is counting how many of them are minority and
                # majority samples
                for k in ind[min_ind[i]][:(j+1)]:
                    if np.all(reg_min <= X[k]) and np.all(X[k] <= reg_max):
                        if y[k] == self.min_label:
                            r_min.append(k)
                        else:
                            r_maj.append(k)

                # appending the coordinates of points to the minority and
                # majority regions
                regions_min.append(r_min)
                regions_maj.append(r_maj)

            # taking the cumulative unions of minority and majority points
            for j in range(1, len(regions_min)):
                regions_min[j] = list(
                    set(regions_min[j]).union(set(regions_min[j-1])))
                regions_maj[j] = list(
                    set(regions_maj[j]).union(set(regions_maj[j-1])))

            # computing the lengths of the increasing minority and majority
            # sets
            regions_min = np.array([len(r) for r in regions_min])
            regions_maj = np.array([len(r) for r in regions_maj])

            # computing the precision of minority classification (all points
            # are supposed to be classified as minority)
            prec = regions_min/(regions_min + regions_maj)
            # taking the difference
            d = np.diff(prec, 1)
            # finding the biggest drop (+1 because diff reduces length, +1
            # because of indexing begins with 0)
            if len(d) == 0:
                k = 0
            else:
                k = np.argmin(d) + 2
            # appending the coordinate of the biggest drop as the ideal
            # neighborhood size note that k indices the minority neighbors
            kappa.append(k)

        # finding nearest minority neighbors of minority samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=max(kappa) + 1, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        ind = nn.kneighbors(X_min, return_distance=False)

        if np.sum(kappa) == 0:
            _logger.warning(self.__class__.__name__ + ": " +
                            "No minority samples in nearest neighbors")
            return X.copy(), y.copy()

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            # choose random point
            idx = self.random_state.randint(len(X_min))
            if kappa[idx] > 0:
                domain = ind[idx][1:(kappa[idx]+1)]
                X_b = X_min[self.random_state.choice(domain)]
                samples.append(self.sample_between_points(X_min[idx], X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'K': self.K,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

