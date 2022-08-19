"""
This module implements the condensed nearest neighbors technique.
"""

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from ._noisefilter import NoiseFilter

from .._logger import logger
_logger= logger

__all__= ['CondensedNearestNeighbors']


class CondensedNearestNeighbors(NoiseFilter):
    """
    Condensed nearest neighbors

    References:
        * BibTex::

            @ARTICLE{condensed_nn,
                        author={Hart, P.},
                        journal={IEEE Transactions on Information Theory},
                        title={The condensed nearest neighbor rule (Corresp.)},
                        year={1968},
                        volume={14},
                        number={3},
                        pages={515-516},
                        keywords={Pattern classification},
                        doi={10.1109/TIT.1968.1054155},
                        ISSN={0018-9448},
                        month={May}}
    """

    def __init__(self, n_jobs=1, **_kwargs):
        """
        Constructor of the noise removing object

        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def get_params(self, deep=False):
        return {'n_jobs': self.n_jobs,
                **NoiseFilter.get_params(self, deep)}

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.array): features
            y (np.array): target labels

        Returns:
            np.array, np.array: dataset after noise removal
        """
        _logger.info("%s: Running noise removal", self.__class__.__name__)

        self.class_label_statistics(y)

        # Initial result set consists of all minority samples and 1 majority
        # sample

        X_maj = X[y == self.maj_label]
        X_hat = np.vstack([X[y == self.min_label], X_maj[0]]) # pylint: disable=invalid-name
        y_hat = np.hstack([np.repeat(self.min_label, len(X_hat)-1),
                           [self.maj_label]])
        X_maj = X_maj[1:]

        # Adding misclassified majority elements repeatedly
        while len(X_maj) != 0:
            knn = KNeighborsClassifier(n_neighbors=1, n_jobs=self.n_jobs)
            knn.fit(X_hat, y_hat)
            pred = knn.predict(X_maj)

            if np.all(pred == self.maj_label):
                break

            X_hat = np.vstack([X_hat, X_maj[pred != self.maj_label]]) # pylint: disable=invalid-name
            y_hat = np.hstack([y_hat,
                                np.repeat(self.maj_label,
                                          len(X_hat) - len(y_hat))])

            X_maj = np.delete(X_maj,
                              np.where(pred != self.maj_label)[0], axis=0)

        return X_hat, y_hat
