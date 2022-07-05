import numpy as np

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from .._base import (StatisticsMixin, ParameterCheckingMixin, 
                        ParameterCombinationsMixin, mode)
from .._metric_tensor import MetricLearningMixin, NearestNeighborsWithMetricTensor

from .._logger import logger
_logger= logger

__all__= ['NoiseFilter',
           'TomekLinkRemoval',
           'CondensedNearestNeighbors',
           'OneSidedSelection',
           'CNNTomekLinks',
           'NeighborhoodCleaningRule',
           'EditedNearestNeighbors']

class NoiseFilter(StatisticsMixin,
                  ParameterCheckingMixin,
                  ParameterCombinationsMixin,
                  MetricLearningMixin):
    """
    Parent class of noise filtering methods
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def remove_noise(self, X, y):
        """
        Removes noise
        Args:
            X (np.array): features
            y (np.array): target labels
        """
        pass

    def get_params(self, deep=False):
        """
        Return parameters

        Returns:
            dict: dictionary of parameters
        """

        return {}

    def set_params(self, **params):
        """
        Set parameters

        Args:
            params (dict): dictionary of parameters
        """

        for key, value in params.items():
            setattr(self, key, value)

        return self


class TomekLinkRemoval(NoiseFilter):
    """
    Tomek link removal

    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
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

    def __init__(self, 
                 strategy='remove_majority', 
                 nn_params={},
                 n_jobs=1):
        """
        Constructor of the noise filter.

        Args:
            strategy (str): noise removal strategy:
                            'remove_majority'/'remove_both'
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_isin(strategy, 'strategy', [
                        'remove_majority', 'remove_both'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.strategy = strategy
        self.nn_params= nn_params
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # using 2 neighbors because the first neighbor is the point itself
        nn= NearestNeighborsWithMetricTensor(n_neighbors=2, 
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        indices= nn.fit(X).kneighbors(X, return_distance=False)

        # identify links
        links = []
        for i in range(len(indices)):
            if indices[indices[i][1]][1] == i:
                if not y[indices[i][1]] == y[indices[indices[i][1]][1]]:
                    links.append((i, indices[i][1]))

        # determine links to be removed
        to_remove = []
        for li in links:
            if self.strategy == 'remove_majority':
                if y[li[0]] == self.min_label:
                    to_remove.append(li[1])
                else:
                    to_remove.append(li[0])
            elif self.strategy == 'remove_both':
                to_remove.append(li[0])
                to_remove.append(li[1])
            else:
                m = 'No Tomek link strategy %s implemented' % self.strategy
                raise ValueError(self.__class__.__name__ + ": " + m)

        to_remove = list(set(to_remove))

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)


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

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removing object

        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise from dataset

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: dataset after noise removal
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        # Initial result set consists of all minority samples and 1 majority
        # sample

        X_maj = X[y == self.maj_label]
        X_hat = np.vstack([X[y == self.min_label], X_maj[0]])
        y_hat = np.hstack([np.repeat(self.min_label, len(X_hat)-1),
                           [self.maj_label]])
        X_maj = X_maj[1:]

        # Adding misclassified majority elements repeatedly
        while True:
            knn = KNeighborsClassifier(n_neighbors=1, n_jobs=self.n_jobs)
            knn.fit(X_hat, y_hat)
            pred = knn.predict(X_maj)

            if np.all(pred == self.maj_label):
                break
            else:
                X_hat = np.vstack([X_hat, X_maj[pred != self.maj_label]])
                y_hat = np.hstack(
                    [y_hat,
                     np.repeat(self.maj_label, len(X_hat) - len(y_hat))])
                X_maj = np.delete(X_maj, np.where(
                    pred != self.maj_label)[0], axis=0)
                if len(X_maj) == 0:
                    break

        return X_hat, y_hat


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

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            n_jobs (int): number of jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        t = TomekLinkRemoval(n_jobs=self.n_jobs)
        X0, y0 = t.remove_noise(X, y)
        cnn = CondensedNearestNeighbors(n_jobs=self.n_jobs)

        return cnn.remove_noise(X0, y0)


class CNNTomekLinks(NoiseFilter):
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

    def __init__(self, n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        c = CondensedNearestNeighbors(n_jobs=self.n_jobs)
        X0, y0 = c.remove_noise(X, y)
        t = TomekLinkRemoval(n_jobs=self.n_jobs)

        return t.remove_noise(X0, y0)


class NeighborhoodCleaningRule(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
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

    def __init__(self, nn_params={}, n_jobs=1):
        """
        Constructor of the noise removal object

        Args:
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.nn_params = nn_params
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        # fitting nearest neighbors with proposed parameter
        # using 4 neighbors because the first neighbor is the point itself
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=4, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)

        # identifying the samples to be removed
        to_remove = []
        for i in range(len(X)):
            if (y[i] == self.maj_label and
                    mode(y[indices[i][1:]]) == self.min_label):
                # if sample i is majority and the decision based on
                # neighbors is minority
                to_remove.append(i)
            elif (y[i] == self.min_label and
                  mode(y[indices[i][1:]]) == self.maj_label):
                # if sample i is minority and the decision based on
                # neighbors is majority
                for j in indices[i][1:]:
                    if y[j] == self.maj_label:
                        to_remove.append(j)

        # removing the noisy samples and returning the results
        to_remove = list(set(to_remove))
        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)


class EditedNearestNeighbors(NoiseFilter):
    """
    References:
        * BibTex::

            @article{smoteNoise0,
                     author = {Batista, Gustavo E. A. P. A. and Prati,
                                Ronaldo C. and Monard, Maria Carolina},
                     title = {A Study of the Behavior of Several Methods for
                                Balancing Machine Learning Training Data},
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

    def __init__(self, 
                 remove='both', 
                 nn_params={},
                 n_jobs=1
                 ):
        """
        Constructor of the noise removal object

        Args:
            remove (str): class to remove from 'both'/'min'/'maj'
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
        """
        super().__init__()

        self.check_isin(remove, 'remove', ['both', 'min', 'maj'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.remove = remove
        self.nn_params= nn_params
        self.n_jobs = n_jobs

    def remove_noise(self, X, y):
        """
        Removes noise

        Args:
            X (np.matrix): features
            y (np.array): target labels

        Returns:
            np.matrix, np.array: cleaned features and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running noise removal via %s" % self.__class__.__name__)
        self.class_label_statistics(X, y)

        if len(X) < 4:
            _logger.info(self.__class__.__name__ + ': ' +
                         "Not enough samples for noise removal")
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=4, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        indices= nn.fit(X).kneighbors(X, return_distance=False)

        to_remove = []
        for i in range(len(X)):
            if not y[i] == mode(y[indices[i][1:]]):
                if (self.remove == 'both' or
                    (self.remove == 'min' and y[i] == self.min_label) or
                        (self.remove == 'maj' and y[i] == self.maj_label)):
                    to_remove.append(i)

        return np.delete(X, to_remove, axis=0), np.delete(y, to_remove)

    def get_params(self):
        """
        Get noise removal parameters

        Returns:
            dict: dictionary of parameters
        """
        return {'remove': self.remove}
