import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SVM_balance']

class SVM_balance(OverSampling):
    """
    References:
        * BibTex::

            @article{svm_balance,
                     author = {Farquad, M.A.H. and Bose, Indranil},
                     title = {Preprocessing Unbalanced Data Using Support
                                Vector Machine},
                     journal = {Decis. Support Syst.},
                     issue_date = {April, 2012},
                     volume = {53},
                     number = {1},
                     month = apr,
                     year = {2012},
                     issn = {0167-9236},
                     pages = {226--233},
                     numpages = {8},
                     url = {http://dx.doi.org/10.1016/j.dss.2012.01.016},
                     doi = {10.1016/j.dss.2012.01.016},
                     acmid = {2181554},
                     publisher = {Elsevier Science Publishers B. V.},
                     address = {Amsterdam, The Netherlands, The Netherlands},
                     keywords = {COIL data, Hybrid method, Preprocessor, SVM,
                                    Unbalanced data},
                    }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_changes_majority,
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
            n_neighbors (int): number of neighbors in the SMOTE sampling
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

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        X, y = SMOTE(proportion=self.proportion,
                     n_neighbors=self.n_neighbors,
                     nn_params=self.nn_params,
                     n_jobs=self.n_jobs,
                     random_state=self._random_state_init).sample(X, y)

        if sum(y == self.min_label) < 2:
            return X.copy(), y.copy()
        else:
            cv = min([5, sum(y == self.min_label)])

        ss = StandardScaler()
        X_norm = ss.fit_transform(X)

        C_params = [0.01, 0.1, 1.0, 10.0]
        best_score = 0
        best_C = 0.01
        for C in C_params:
            _logger.info(self.__class__.__name__ + ": " +
                         "Evaluating SVM with C=%f" % C)
            svc = SVC(C=C, kernel='rbf', gamma='auto')
            score = np.mean(cross_val_score(svc, X_norm, y, cv=cv))
            if score > best_score:
                best_score = score
                best_C = C
        svc = SVC(C=best_C, kernel='rbf', gamma='auto')
        svc.fit(X_norm, y)

        return X, svc.predict(X_norm)

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
