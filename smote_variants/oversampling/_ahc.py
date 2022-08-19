"""
This module implements the AHC method
"""
import warnings

import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans

from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['AHC']

class AHC(OverSampling):
    """
    References:
        * BibTex::

            @article{AHC,
                    title = "Learning from imbalanced data in surveillance
                             of nosocomial infection",
                    journal = "Artificial Intelligence in Medicine",
                    volume = "37",
                    number = "1",
                    pages = "7 - 18",
                    year = "2006",
                    note = "Intelligent Data Analysis in Medicine",
                    issn = "0933-3657",
                    doi = "https://doi.org/10.1016/j.artmed.2005.03.002",
                    url = {http://www.sciencedirect.com/science/article/
                            pii/S0933365705000850},
                    author = "Gilles Cohen and Mélanie Hilario and Hugo Sax
                                and Stéphane Hugonnet and Antoine Geissbuhler",
                    keywords = "Nosocomial infection, Machine learning,
                                    Support vector machines, Data imbalance"
                    }
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_application]

    def __init__(self,
                 *,
                 strategy='min',
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            strategy (str): which class to sample (min/maj/minmaj)
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_isin(strategy, 'strategy', ['min', 'maj', 'minmaj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'strategy': ['min', 'maj', 'minmaj']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample_majority(self, X, n_clusters):
        """
        Sample the majority class

        Args:
            X (np.ndarray): majority samples
            n_clusters (int): number of clusters to find

        Returns:
            np.ndarray: downsampled vectors
        """
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self._random_state_init)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kmeans.fit(X)
        return kmeans.cluster_centers_

    def sample_minority(self, X):
        """
        Sampling the minority class

        Args:
            X (np.ndarray): minority samples

        Returns:
            np.ndarray: the oversampled set of vectors
        """
        aggc = AgglomerativeClustering(n_clusters=1)
        aggc.fit(X)
        n_samples = len(X)

        clus_cent = [None]*len(aggc.children_)
        weights = [None]*len(aggc.children_)

        def cluster_centers(children, idx, clus_cent, weights):
            """
            Extract cluster centers

            Args:
                children (np.array): indices of children
                idx (int): index to process
                clus_cent (np.array): cluster centers
                weights (np.array): cluster weights

            Returns:
                int, float: new cluster center, new weight
            """
            if idx < n_samples:
                return X[idx], 1.0

            if clus_cent[idx - n_samples] is None:
                a_cent, w_a = cluster_centers(
                    children, children[idx - n_samples][0], clus_cent, weights)
                b_cent, w_b = cluster_centers(
                    children, children[idx - n_samples][1], clus_cent, weights)
                clus_cent[idx - n_samples] = (w_a*a_cent + w_b*b_cent)/(w_a + w_b)
                weights[idx - n_samples] = w_a + w_b

            return clus_cent[idx - n_samples], weights[idx - n_samples]

        cluster_centers(aggc.children_, aggc.children_[-1][-1] + 1, clus_cent, weights)

        return np.vstack(clus_cent)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        # extracting minority samples
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        if self.strategy == 'maj':
            X_maj_resampled = self.sample_majority(X_maj, len(X_min)) # pylint: disable=invalid-name
            return (np.vstack([X_min, X_maj_resampled]),
                    np.hstack([np.repeat(self.min_label, len(X_min)),
                               np.repeat(self.maj_label,
                                         len(X_maj_resampled))]))
        if self.strategy == 'min':
            X_min_resampled = self.sample_minority(X_min) # pylint: disable=invalid-name
            return (np.vstack([X_min_resampled, X_min, X_maj]),
                    np.hstack([np.repeat(self.min_label,
                                         (len(X_min_resampled) + len(X_min))),
                               np.repeat(self.maj_label, len(X_maj))]))
        # the "minmaj" strategy case
        X_min_resampled = self.sample_minority(X_min) # pylint: disable=invalid-name
        n_maj_sample = min([len(X_maj), len(X_min_resampled) + len(X_min)])
        X_maj_resampled = self.sample_majority(X_maj, n_maj_sample) # pylint: disable=invalid-name
        return (np.vstack([X_min_resampled, X_min, X_maj_resampled]),
                np.hstack([np.repeat(self.min_label,
                                        (len(X_min_resampled) + len(X_min))),
                            np.repeat(self.maj_label,
                                        len(X_maj_resampled))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'strategy': self.strategy,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
