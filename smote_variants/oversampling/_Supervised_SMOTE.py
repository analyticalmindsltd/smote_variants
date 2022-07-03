import numpy as np

from sklearn.ensemble import RandomForestClassifier

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['Supervised_SMOTE']

class Supervised_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @article{supervised_smote,
                        author = {Hu, Jun AND He, Xue AND Yu, Dong-Jun AND
                                    Yang, Xi-Bei AND Yang, Jing-Yu AND Shen,
                                    Hong-Bin},
                        journal = {PLOS ONE},
                        publisher = {Public Library of Science},
                        title = {A New Supervised Over-Sampling Algorithm
                                    with Application to Protein-Nucleotide
                                    Binding Residue Prediction},
                        year = {2014},
                        month = {09},
                        volume = {9},
                        url = {https://doi.org/10.1371/journal.pone.0107676},
                        pages = {1-10},
                        number = {9},
                        doi = {10.1371/journal.pone.0107676}
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_application]

    def __init__(self,
                 proportion=1.0,
                 *,
                 th_lower=0.5,
                 th_upper=1.0,
                 classifier=RandomForestClassifier(n_estimators=50,
                                                   n_jobs=1,
                                                   random_state=5),
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            th_lower (float): lower bound of the confidence interval
            th_upper (float): upper bound of the confidence interval
            classifier (obj): classifier used to estimate class memberships
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_in_range(th_lower, "th_lower", [0, 1])
        self.check_in_range(th_upper, "th_upper", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.th_lower = th_lower
        self.th_upper = th_upper
        self.classifier = classifier
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        classifiers = [RandomForestClassifier(n_estimators=50,
                                              n_jobs=1,
                                              random_state=5)]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'th_lower': [0.3, 0.5, 0.8],
                                  'th_upper': [1.0],
                                  'classifier': classifiers}
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

        # training the classifier
        self.classifier.fit(X, y)

        X_min = X[y == self.min_label]

        th_lower = self.th_lower

        # do the sampling
        samples = []
        n_trials = 1
        n_success = 1
        while len(samples) < n_to_sample:
            n_trials = n_trials + 1

            domain = range(len(X_min))
            x0, x1 = self.random_state.choice(domain, 2, replace=False)
            x0, x1 = X_min[x0], X_min[x1]
            sample = self.sample_between_points(x0, x1)
            probs = self.classifier.predict_proba(sample.reshape(1, -1))
            # extract probability
            class_column = np.where(self.classifier.classes_ == self.min_label)
            class_column = class_column[0][0]
            prob = probs[0][class_column]
            if prob >= th_lower and prob <= self.th_upper:
                samples.append(sample)
                n_success = n_success + 1

            # decreasing lower threshold if needed
            if n_success/n_trials < 0.02:
                th_lower = th_lower * 0.9
                n_success = 1
                n_trials = 1

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'th_lower': self.th_lower,
                'th_upper': self.th_upper,
                'classifier': self.classifier,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

