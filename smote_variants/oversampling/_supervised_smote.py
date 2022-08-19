"""
This module implements the Supervised_SMOTE method.
"""

import numpy as np

from scipy.linalg import circulant

from ..base import OverSamplingSimplex
from ..base import instantiate_obj, coalesce_dict

from .._logger import logger
_logger= logger

__all__= ['Supervised_SMOTE']

class Supervised_SMOTE(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_uses_classifier,
                  OverSamplingSimplex.cat_application]

    def __init__(self,
                 proportion=1.0,
                 *,
                 th_lower=0.5,
                 th_upper=1.0,
                 classifier=('sklearn.ensemble',
                                'RandomForestClassifier',
                                {'n_estimators': 50,
                                'n_jobs': 1,
                                'random_state': 5}),
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
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
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                    'within_simplex_sampling': 'random',
                    'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_in_range(th_lower, "th_lower", [0, 1])
        self.check_in_range(th_upper, "th_upper", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.th_lower = th_lower
        self.th_upper = th_upper
        self.classifier = classifier
        self.classifier_obj = instantiate_obj(classifier)
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        classifiers = [('sklearn.ensemble',
                        'RandomForestClassifier',
                        {'n_estimators': 50,
                        'n_jobs': 1,
                        'random_state': 5})]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'th_lower': [0.3, 0.5, 0.8],
                                  'th_upper': [1.0],
                                  'classifier': classifiers}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # training the classifier
        self.classifier_obj.fit(X, y)

        X_min = X[y == self.min_label]

        th_lower = self.th_lower

        indices = circulant(np.arange(X_min.shape[0]))

        class_column = np.where(self.classifier_obj.classes_ == self.min_label)
        class_column = class_column[0][0]

        samples = np.zeros(shape=(0, X.shape[1]))
        n_trials = 1
        n_success = 1
        while len(samples) < n_to_sample:
            to_sample = np.max([n_to_sample - len(samples), 10])

            n_trials = n_trials + to_sample

            samples_tmp = self.sample_simplex(X=X_min,
                                          indices=indices,
                                          n_to_sample=to_sample)

            prob = self.classifier_obj.predict_proba(samples_tmp)
            prob = prob[:, class_column]

            samples_tmp = samples_tmp[(prob >= th_lower)
                                    & (prob <= self.th_upper)]

            samples = np.vstack([samples, samples_tmp])

            n_success = n_success + samples_tmp.shape[0]

            if n_success/n_trials < 0.02:
                th_lower = th_lower * 0.9
                n_success = 1
                n_trials = 1

        samples = samples[:np.min([samples.shape[0], n_to_sample])]

        return (np.vstack([X, samples]),
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
                **OverSamplingSimplex.get_params(self)}
