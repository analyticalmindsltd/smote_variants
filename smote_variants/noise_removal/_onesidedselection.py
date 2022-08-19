"""
This module implements the one sided selection technique.
"""

from ._noisefilter import NoiseFilter
from ._condensednn import CondensedNearestNeighbors
from ._tomeklinkremoval import TomekLinkRemoval

from .._logger import logger
_logger= logger

__all__= ['OneSidedSelection']

class OneSidedSelection(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods
                                for Balancing Machine Learning Training Data},
                     journal = {SIGKDD Explor. Newsl.},
                     issue_date = {June 2004},
                     volume = {6},
                     number = {1},
                     month = jun,
                     year = {2004},
                     issn = {1931-0145},
                     pages = {20--29},
                     numpages = {10},
                     url = {http://doi.acm.org/10.1145/1007730.1007735},
                     doi = {10.1145/1007730.1007735},
                     acmid = {1007735},
                     publisher = {ACM},
                     address = {New York, NY, USA}
                    }
    """

    def __init__(self, n_jobs=1, **_kwargs):
        """
        Constructor of the noise removal object

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
        Removes noise

        Args:
            X (np.array): features
            y (np.array): target labels

        Returns:
            np.array, np.array: cleaned features and target labels
        """
        _logger.info("%s: Running noise removal.", self.__class__.__name__)
        self.class_label_statistics(y)

        tomek = TomekLinkRemoval(n_jobs=self.n_jobs)
        X_tomek, y_tomek = tomek.remove_noise(X, y) # pylint: disable=invalid-name
        cnn = CondensedNearestNeighbors(n_jobs=self.n_jobs)

        return cnn.remove_noise(X_tomek, y_tomek)
