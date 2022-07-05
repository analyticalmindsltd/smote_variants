import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

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
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            strategy (str): which class to sample (min/maj/minmaj)
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_isin(strategy, 'strategy', ['min', 'maj', 'minmaj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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
        ac = AgglomerativeClustering(n_clusters=1)
        ac.fit(X)
        n_samples = len(X)

        cc = [None]*len(ac.children_)
        weights = [None]*len(ac.children_)

        def cluster_centers(children, i, cc, weights):
            """
            Extract cluster centers

            Args:
                children (np.array): indices of children
                i (int): index to process
                cc (np.array): cluster centers
                weights (np.array): cluster weights

            Returns:
                int, float: new cluster center, new weight
            """
            if i < n_samples:
                return X[i], 1.0

            if cc[i - n_samples] is None:
                a, w_a = cluster_centers(
                    children, children[i - n_samples][0], cc, weights)
                b, w_b = cluster_centers(
                    children, children[i - n_samples][1], cc, weights)
                cc[i - n_samples] = (w_a*a + w_b*b)/(w_a + w_b)
                weights[i - n_samples] = w_a + w_b

            return cc[i - n_samples], weights[i - n_samples]

        cluster_centers(ac.children_, ac.children_[-1][-1] + 1, cc, weights)

        return np.vstack(cc)

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

        # extracting minority samples
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        if self.strategy == 'maj':
            X_maj_resampled = self.sample_majority(X_maj, len(X_min))
            return (np.vstack([X_min, X_maj_resampled]),
                    np.hstack([np.repeat(self.min_label, len(X_min)),
                               np.repeat(self.maj_label,
                                         len(X_maj_resampled))]))
        elif self.strategy == 'min':
            X_min_resampled = self.sample_minority(X_min)
            return (np.vstack([X_min_resampled, X_min, X_maj]),
                    np.hstack([np.repeat(self.min_label,
                                         (len(X_min_resampled) + len(X_min))),
                               np.repeat(self.maj_label, len(X_maj))]))
        elif self.strategy == 'minmaj':
            X_min_resampled = self.sample_minority(X_min)
            n_maj_sample = min([len(X_maj), len(X_min_resampled) + len(X_min)])
            X_maj_resampled = self.sample_majority(X_maj, n_maj_sample)
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
                'random_state': self._random_state_init}
