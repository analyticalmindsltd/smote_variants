import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_IPF']

class SMOTE_IPF(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_ipf,
                        title = "SMOTE–IPF: Addressing the noisy and borderline
                                    examples problem in imbalanced
                                    classification by a re-sampling method
                                    with filtering",
                        journal = "Information Sciences",
                        volume = "291",
                        pages = "184 - 203",
                        year = "2015",
                        issn = "0020-0255",
                        doi = "https://doi.org/10.1016/j.ins.2014.08.051",
                        author = "José A. Sáez and Julián Luengo and Jerzy
                                    Stefanowski and Francisco Herrera",
                        keywords = "Imbalanced classification,
                                        Borderline examples,
                                        Noisy data,
                                        Noise filters,
                                        SMOTE"
                        }
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_folds=9,
                 k=3,
                 p=0.01,
                 voting='majority',
                 classifier=DecisionTreeClassifier(random_state=2),
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): number of neighbors in SMOTE sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_folds (int): the number of partitions
            k (int): used in stopping condition
            p (float): percentage value ([0,1]) used in stopping condition
            voting (str): 'majority'/'consensus'
            classifier (obj): classifier object
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(n_folds, "n_folds", 2)
        self.check_greater_or_equal(k, "k", 1)
        self.check_greater_or_equal(p, "p", 0)
        self.check_isin(voting, "voting", ['majority', 'consensus'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_folds = n_folds
        self.k = k
        self.p = p
        self.voting = voting
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
        classifiers = [DecisionTreeClassifier(random_state=2)]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'n_folds': [9],
                                  'k': [3],
                                  'p': [0.01],
                                  'voting': ['majority', 'consensus'],
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

        # do SMOTE sampling
        X_samp, y_samp = SMOTE(proportion=self.proportion,
                               n_neighbors=self.n_neighbors,
                               nn_params=self.nn_params,
                               n_jobs=self.n_jobs,
                               random_state=self._random_state_init).sample(X, y)

        n_folds = min([self.n_folds, np.sum(y == self.min_label)])

        condition = 0
        while True:
            # validating the sampled dataset
            validator = StratifiedKFold(n_folds)
            predictions = []
            for train_index, _ in validator.split(X_samp, y_samp):
                self.classifier.fit(X_samp[train_index], y_samp[train_index])
                predictions.append(self.classifier.predict(X_samp))

            # do decision based on one of the voting schemes
            if self.voting == 'majority':
                pred_votes = (np.mean(predictions, axis=0) > 0.5).astype(int)
                to_remove = np.where(np.not_equal(pred_votes, y_samp))[0]
            elif self.voting == 'consensus':
                pred_votes = (np.mean(predictions, axis=0) > 0.5).astype(int)
                sum_votes = np.sum(predictions, axis=0)
                to_remove = np.where(np.logical_and(np.not_equal(
                    pred_votes, y_samp), np.equal(sum_votes, self.n_folds)))[0]
            else:
                message = 'Voting scheme %s is not implemented' % self.voting
                raise ValueError(self.__class__.__name__ + ": " + message)

            # delete samples incorrectly classified
            _logger.info(self.__class__.__name__ + ": " +
                         'Removing %d elements' % len(to_remove))
            X_samp = np.delete(X_samp, to_remove, axis=0)
            y_samp = np.delete(y_samp, to_remove)

            # if the number of samples removed becomes small or k iterations
            # were done quit
            if len(to_remove) < len(X_samp)*self.p:
                condition = condition + 1
            else:
                condition = 0
            if condition >= self.k:
                break

        return X_samp, y_samp

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_folds': self.n_folds,
                'k': self.k,
                'p': self.p,
                'voting': self.voting,
                'n_jobs': self.n_jobs,
                'classifier': self.classifier,
                'random_state': self._random_state_init}
