import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['NRAS']

class NRAS(OverSampling):
    """
    References:
        * BibTex::

            @article{nras,
                        title = "Noise Reduction A Priori Synthetic
                                    Over-Sampling for class imbalanced data
                                    sets",
                        journal = "Information Sciences",
                        volume = "408",
                        pages = "146 - 161",
                        year = "2017",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2017.04.046",
                        author = "William A. Rivera",
                        keywords = "NRAS, SMOTE, OUPS, Class imbalance,
                                        Classification"
                        }
    """

    categories = [OverSampling.cat_sample_ordinary,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 t=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            t (float): [0,1] fraction of n_neighbors as threshold
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(t, "t", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.t = t
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
                                  'n_neighbors': [5, 7, 9],
                                  't': [0.3, 0.5, 0.8]}
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

        # standardization is needed to make the range of the propensity scores
        # similar to that of the features
        mms = MinMaxScaler()
        X_trans = mms.fit_transform(X)

        # determining propensity scores using logistic regression
        lr = LogisticRegression(solver='lbfgs',
                                n_jobs=self.n_jobs,
                                random_state=self._random_state_init)
        lr.fit(X_trans, y)
        propensity = lr.predict_proba(X_trans)[:, np.where(
            lr.classes_ == self.min_label)[0][0]]

        X_min = X_trans[y == self.min_label]

        # adding propensity scores as a new feature
        X_new = np.column_stack([X_trans, propensity])
        X_min_new = X_new[y == self.min_label]

        # finding nearest neighbors of minority samples
        n_neighbors = min([len(X_new), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X_new, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_new)
        ind = nn.kneighbors(X_min_new, return_distance=False)

        # do the sampling
        samples = []
        to_remove = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            # finding the number of minority neighbors
            t_hat = np.sum(y[ind[idx][1:]] == self.min_label)
            if t_hat < self.t*n_neighbors:
                # removing the minority point if the number of minority
                # neighbors is less then the threshold
                # to_remove indexes X_min
                if idx not in to_remove:
                    to_remove.append(idx)
                    # compensating the removal of the minority point
                    n_to_sample = n_to_sample + 1

                if len(to_remove) == len(X_min):
                    _logger.warning(self.__class__.__name__ + ": " +
                                    "all minority samples identified as noise")
                    return X.copy(), y.copy()
            else:
                # otherwise do the sampling
                X_b = X_trans[self.random_state.choice(ind[idx][1:])]
                samples.append(self.sample_between_points(X_min[idx], X_b))

        # remove noisy elements
        X_maj = X_trans[y == self.maj_label]
        X_min = np.delete(X_min, to_remove, axis=0)

        return (mms.inverse_transform(np.vstack([X_maj,
                                                 X_min,
                                                 np.vstack(samples)])),
                np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min)),
                           np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                't': self.t,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
