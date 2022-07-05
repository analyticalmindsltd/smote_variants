import numpy as np

from sklearn.metrics import pairwise_distances

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_RSB']

class SMOTE_RSB(OverSampling):
    """
    References:
        * BibTex::

            @Article{smote_rsb,
                    author="Ramentol, Enislay
                    and Caballero, Yail{\'e}
                    and Bello, Rafael
                    and Herrera, Francisco",
                    title="SMOTE-RSB*: a hybrid preprocessing approach
                            based on oversampling and undersampling for
                            high imbalanced data-sets using SMOTE and
                            rough sets theory",
                    journal="Knowledge and Information Systems",
                    year="2012",
                    month="Nov",
                    day="01",
                    volume="33",
                    number="2",
                    pages="245--265",
                    issn="0219-3116",
                    doi="10.1007/s10115-011-0465-6",
                    url="https://doi.org/10.1007/s10115-011-0465-6"
                    }

    Notes:
        * I think the description of the algorithm in Fig 5 of the paper
            is not correct. The set "resultSet" is initialized with the
            original instances, and then the While loop in the Algorithm
            run until resultSet is empty, which never holds. Also, the
            resultSet is only extended in the loop. Our implementation
            is changed in the following way: we generate twice as many
            instances are required to balance the dataset, and repeat
            the loop until the number of new samples added to the training
            set is enough to balance the dataset.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
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
                                the number of minority samples will be equal
                                to the number of majority samples
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

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
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

        X_maj = X[y == self.maj_label]
        X_min = X[y == self.min_label]

        # Step 1: do the sampling
        smote = SMOTE(proportion=self.proportion,
                      n_neighbors=self.n_neighbors,
                      nn_params=self.nn_params,
                      n_jobs=self.n_jobs,
                      random_state=self._random_state_init)

        X_samp, y_samp = smote.sample(X, y)
        X_samp, y_samp = X_samp[len(X):], y_samp[len(X):]

        if len(X_samp) == 0:
            return X.copy(), y.copy()

        # Step 2: (original will be added later)
        result_set = []

        # Step 3: first the data is normalized
        maximums = np.max(X_samp, axis=0)
        minimums = np.min(X_samp, axis=0)

        # normalize X_new and X_maj
        norm_factor = maximums - minimums
        null_mask = norm_factor == 0
        n_null = np.sum(null_mask)
        fixed = np.max(np.vstack([maximums[null_mask], np.repeat(1, n_null)]),
                       axis=0)

        norm_factor[null_mask] = fixed

        X_samp_norm = X_samp / norm_factor
        X_maj_norm = X_maj / norm_factor

        # compute similarity matrix
        similarity_matrix = 1.0 - pairwise_distances(X_samp_norm,
                                                     X_maj_norm,
                                                     metric='minkowski',
                                                     p=1)/len(X[0])

        # Step 4: counting the similar examples
        similarity_value = 0.4
        syn = len(X_samp)
        cont = np.zeros(syn)

        already_added = np.repeat(False, len(X_samp))

        while (len(result_set) < len(X_maj) - len(X_min)
                and similarity_value <= 0.9):
            for i in range(syn):
                cont[i] = np.sum(similarity_matrix[i, :] > similarity_value)
                if cont[i] == 0 and not already_added[i]:
                    result_set.append(X_samp[i])
                    already_added[i] = True
            similarity_value = similarity_value + 0.05

        # Step 5: returning the results depending the number of instances
        # added to the result set
        if len(result_set) > 0:
            return (np.vstack([X, np.vstack(result_set)]),
                    np.hstack([y, np.repeat(self.min_label,
                                            len(result_set))]))
        else:
            return np.vstack([X, X_samp]), np.hstack([y, y_samp])

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
