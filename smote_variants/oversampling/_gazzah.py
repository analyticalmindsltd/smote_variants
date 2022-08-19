"""
This module implements the Gazzah method.
"""

import numpy as np

from sklearn.decomposition import PCA

from ..base import OverSampling
from ._polynom_fit_smote import polynom_fit_SMOTE_star

from .._logger import logger
_logger= logger

__all__= ['Gazzah']

class Gazzah(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{gazzah,
                            author={Gazzah, S. and Hechkel, A. and Essoukri
                                        Ben Amara, N. },
                            booktitle={2015 IEEE 12th International
                                        Multi-Conference on Systems,
                                        Signals Devices (SSD15)},
                            title={A hybrid sampling method for
                                    imbalanced data},
                            year={2015},
                            volume={},
                            number={},
                            pages={1-6},
                            keywords={computer vision;image classification;
                                        learning (artificial intelligence);
                                        sampling methods;hybrid sampling
                                        method;imbalanced data;
                                        diversification;computer vision
                                        domain;classical machine learning
                                        systems;intraclass variations;
                                        system performances;classification
                                        accuracy;imbalanced training data;
                                        training data set;over-sampling;
                                        minority class;SMOTE star topology;
                                        feature vector deletion;intra-class
                                        variations;distribution criterion;
                                        biometric data;true positive rate;
                                        Training data;Principal component
                                        analysis;Databases;Support vector
                                        machines;Training;Feature extraction;
                                        Correlation;Imbalanced data sets;
                                        Intra-class variations;Data analysis;
                                        Principal component analysis;
                                        One-against-all SVM},
                            doi={10.1109/SSD.2015.7348093},
                            ISSN={},
                            month={March}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_changes_majority]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_components=2,
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
            n_components (int): number of components in PCA analysis
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_components, "n_components", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_components': [2, 3, 4, 5]}
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
        self.class_label_statistics(y)

        # do the oversampling
        pf_smote = polynom_fit_SMOTE_star(proportion=self.proportion,
                                     random_state=self._random_state_init)
        X_samp, _ = pf_smote.sample(X, y)
        X_min_samp = X_samp[len(X):] # pylint: disable=invalid-name

        if len(X_min_samp) == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # do the undersampling
        X_maj = X[y == self.maj_label]

        # fitting the PCA model
        pca = PCA(n_components=min([len(X[0]), self.n_components]))
        X_maj_trans = pca.fit_transform(X_maj) # pylint: disable=invalid-name
        R = np.sqrt(np.sum(np.var(X_maj_trans, axis=0))) # pylint: disable=invalid-name

        # determining the majority samples to remove
        to_remove = np.where(np.linalg.norm(X_maj_trans, axis=1) > R)[0]

        _logger.info("%s: Removing %d majority samples",
                        self.__class__.__name__, len(to_remove))

        # removing the majority samples
        X_maj = np.delete(X_maj, to_remove, axis=0)

        return (np.vstack([X_maj, X_min_samp]),
                np.hstack([np.repeat(self.maj_label, len(X_maj)),
                           np.repeat(self.min_label, len(X_min_samp))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_components': self.n_components,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
