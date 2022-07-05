import numpy as np

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['RWO_sampling']

class RWO_sampling(OverSampling):
    """
    References:
        * BibTex::

            @article{rwo_sampling,
                    author = {Zhang, Huaxzhang and Li, Mingfang},
                    year = {2014},
                    month = {11},
                    pages = {},
                    title = {RWO-Sampling: A Random Walk Over-Sampling Approach
                                to Imbalanced Data Classification},
                    volume = {20},
                    booktitle = {Information Fusion}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self, 
                 proportion=1.0,
                 *, 
                 n_jobs=1, 
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
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
                                                 1.0, 1.5, 2.0]}
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

        stds = np.diag(np.std(X_min, axis=0)/np.sqrt(len(X_min)))

        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X_min))
            samples.append(self.random_state.multivariate_normal(X_min[idx],
                                                                 stds))

        return (np.vstack([X, samples]),
                np.hstack([y, np.array([self.min_label]*len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
