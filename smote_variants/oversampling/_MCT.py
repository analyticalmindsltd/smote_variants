import numpy as np

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['MCT']

class MCT(OverSampling):
    """
    References:
        * BibTex::

            @article{mct,
                    author = {Jiang, Liangxiao and Qiu, Chen and Li, Chaoqun},
                    year = {2015},
                    month = {03},
                    pages = {1551004},
                    title = {A Novel Minority Cloning Technique for
                                Cost-Sensitive Learning},
                    volume = {29},
                    booktitle = {International Journal of Pattern Recognition
                                    and Artificial Intelligence}
                    }

    Notes:
        * Mode is changed to median, distance is changed to Euclidean to
                support continuous features, and normalized.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_copy]

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

        # having continuous variables, the mode is replaced by median
        x_med = np.median(X_min, axis=0)
        distances = np.array([np.linalg.norm(x_med - x) for x in X_min])
        sums = np.sum(distances)
        if sums != 0:
            distances = distances/sums

        # distribution of copies is determined (Euclidean distance is a
        # dissimilarity measure which is changed to similarity by subtracting
        # from 1.0)
        distribution = (1.0 - distances)/(np.sum(1.0 - distances))

        if any(np.isnan(distribution)):
            _logger.warning(self.__class__.__name__ + ": " +
                            "NaN in the probability distribution")
            return X.copy(), y.copy()

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            samples.append(X_min[self.random_state.choice(
                np.arange(len(X_min)), p=distribution)])

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
