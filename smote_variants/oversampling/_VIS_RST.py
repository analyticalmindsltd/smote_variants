import numpy as np

from sklearn.preprocessing import StandardScaler

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['VIS_RST']

class VIS_RST(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{vis_rst,
                            author="Borowska, Katarzyna
                            and Stepaniuk, Jaroslaw",
                            editor="Saeed, Khalid
                            and Homenda, Wladyslaw",
                            title="Imbalanced Data Classification: A Novel
                                    Re-sampling Approach Combining Versatile
                                    Improved SMOTE and Rough Sets",
                            booktitle="Computer Information Systems and
                                        Industrial Management",
                            year="2016",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="31--42",
                            isbn="978-3-319-45378-1"
                            }

    Notes:
        * Replication of DANGER samples will be removed by the last step of
            noise filtering.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
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
            n_neighbors (int): number of neighbors
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
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
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
                                  'n_neighbors': [3, 5, 7]}
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

        # standardizing the data
        ss = StandardScaler()
        ss.fit(X)
        X = ss.transform(X)
        y = y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model to determine boundary region
        n_neighbors = min([len(X), self.n_neighbors + 1])
        
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        ind = nn.kneighbors(X_maj, return_distance=False)

        # determining boundary region of majority samples
        boundary = np.array([np.sum(y[ind[i]] == self.maj_label)
                             != n_neighbors for i in range(len(X_maj))])
        y_maj = y[y == self.maj_label]
        y_maj[boundary] = self.min_label
        y[y == self.maj_label] = y_maj

        # extracting new minority and majority set
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # labeling minority samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        ind = nn.kneighbors(X_min, return_distance=False)

        # extracting labels
        labels = []
        for i in range(len(ind)):
            min_class_neighbors = np.sum(y[ind[i][1:]] == self.maj_label)
            if min_class_neighbors == n_neighbors-1:
                labels.append('noise')
            elif min_class_neighbors < n_neighbors/2:
                labels.append('safe')
            else:
                labels.append('danger')
                
        if len(np.unique(labels)) == 1 and labels[0] == 'noise':
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # extracting the number of different labels (noise is not used)
        safe = np.sum([li == 'safe' for li in labels])
        danger = np.sum([li == 'danger' for li in labels])

        if safe == 0:
            mode = 'no_safe'
        elif danger > 0.3*len(X_min):
            mode = 'high_complexity'
        else:
            mode = 'low_complexity'

        # fitting nearest neighbors to find the neighbors of minority elements
        # among minority elements
        n_neighbors_min = min([len(X_min), self.n_neighbors + 1])
        nn_min = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors_min, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
        nn_min.fit(X_min)
        ind_min = nn_min.kneighbors(X_min, return_distance=False)

        # do the sampling
        samples = []
        mask = np.repeat(False, len(X_min))
        while len(samples) < n_to_sample:
            
            # choosing a random minority sample
            idx = self.random_state.choice(np.arange(len(X_min)))
            
            #print(idx, len(samples), n_to_sample, np.unique(labels, return_counts=True))

            # implementation of sampling rules depending on the mode
            # the rules are not covering all cases
            if mode == 'high_complexity':
                if labels[idx] == 'noise':
                    pass
                #elif (labels[idx] == 'danger' and not mask[idx]) or (np.sum(mask) == len(mask)):
                elif labels[idx] == 'danger':
                    samples.append(X_min[idx])
                    mask[idx] = True
                else:
                    X_b = X_min[self.random_state.choice(ind_min[idx][1:])]
                    samples.append(self.sample_between_points(X_min[idx], X_b))
            elif mode == 'low_complexity':
                if labels[idx] == 'noise':
                    pass
                elif labels[idx] == 'danger':
                    X_b = X_min[self.random_state.choice(ind_min[idx][1:])]
                    samples.append(self.sample_between_points(X_min[idx], X_b))
                #elif (not mask[idx]) or (np.sum(mask) == len(mask)):
                else:
                    # tweak (sum == len), otherwise the method falls in infinite loop when
                    # all samples are safe
                    samples.append(X_min[idx])
                    mask[idx] = True
            else:
                X_b = X_min[self.random_state.choice(ind_min[idx][1:])]
                samples.add(self.sample_between_points(X_min[idx], X_b))

        X_samp = np.vstack(samples)

        # final noise removal by removing those minority samples generated
        # and not belonging to the lower approximation
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X)
        ind_check = nn.kneighbors(X_samp, return_distance=False)

        def maj_zero(i):
            return np.sum(y[ind_check[i][1:]] == self.maj_label) == 0

        num_maj_mask = np.array([maj_zero(i) for i in range(len(samples))])
        X_samp = X_samp[num_maj_mask]

        return (ss.inverse_transform(np.vstack([X, X_samp])),
                np.hstack([y, np.repeat(self.min_label, len(X_samp))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
