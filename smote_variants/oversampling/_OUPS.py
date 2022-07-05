import numpy as np

from sklearn.linear_model import LogisticRegression

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['OUPS']

class OUPS(OverSampling):
    """
    References:
        * BibTex::

            @article{oups,
                        title = "A priori synthetic over-sampling methods for
                                    increasing classification sensitivity in
                                    imbalanced data sets",
                        journal = "Expert Systems with Applications",
                        volume = "66",
                        pages = "124 - 135",
                        year = "2016",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2016.09.010",
                        author = "William A. Rivera and Petros Xanthopoulos",
                        keywords = "SMOTE, OUPS, Class imbalance,
                                    Classification"
                        }

    Notes:
        * In the description of the algorithm a fractional number p (j) is
            used to index a vector.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

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

        if self.class_stats[self.min_label] < 2:
            message = ("The number of minority samples (%d) is not enough for"
                       " sampling")
            message = message % self.class_stats[self.min_label]
            _logger.warning(self.__class__.__name__ + ": " + message)
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # extracting propensity scores
        lr = LogisticRegression(solver='lbfgs',
                                n_jobs=self.n_jobs,
                                random_state=self._random_state_init)
        lr.fit(X, y)
        propensity = lr.predict_proba(X)
        propensity = propensity[:, np.where(
            lr.classes_ == self.min_label)[0][0]]

        # sorting indices according to propensity scores
        prop_sorted = sorted(zip(propensity, np.arange(
            len(propensity))), key=lambda x: -x[0])

        p = np.sum(y == self.maj_label)/np.sum(y == self.min_label)
        n = 0
        samples = []
        # implementing Algorithm 1 in the cited paper with some minor changes
        # to enable the proper sampling of p numbers
        while n < len(propensity) and len(samples) < n_to_sample:
            if (y[prop_sorted[n][1]] == self.min_label
                    and n < len(propensity) - 1):
                num = 1
                p_tmp = p
                while p_tmp > 0 and n + num < len(propensity):
                    if self.random_state.random_sample() < p_tmp:
                        samples.append(self.sample_between_points(
                            X[prop_sorted[n][1]], X[prop_sorted[n+num][1]]))
                    p_tmp = p_tmp - 1
                    num = num + 1
            n = n + 1

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
