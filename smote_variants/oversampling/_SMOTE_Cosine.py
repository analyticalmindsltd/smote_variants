import numpy as np

from sklearn.neighbors import NearestNeighbors

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE_Cosine']

class SMOTE_Cosine(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                      title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE:
                                An enhancement strategy to handle imbalance
                                in data level},
                      author={Fajri Koto},
                      journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                      year={2014},
                      pages={280-284}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
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

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        # Fitting the nearest neighbors models to the minority and
        # majority data using two different metrics for the minority
        nn_min_euc = NearestNeighbors(n_neighbors=len(X_min),
                                      n_jobs=self.n_jobs)
        nn_min_euc.fit(X_min)
        nn_min_euc_ind = nn_min_euc.kneighbors(X_min, return_distance=False)

        nn_min_cos = NearestNeighbors(n_neighbors=len(X_min),
                                      metric='cosine',
                                      n_jobs=self.n_jobs)
        nn_min_cos.fit(X_min)
        nn_min_cos_ind = nn_min_cos.kneighbors(X_min, return_distance=False)

        nn_maj = NearestNeighbors(n_neighbors=self.n_neighbors,
                                  n_jobs=self.n_jobs)
        nn_maj.fit(X_maj)
        nn_maj_ind = nn_maj.kneighbors(X_min, return_distance=False)

        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.choice(
                np.arange(len(minority_indices)))
            u = X[minority_indices[random_idx]]
            # get the rank of each minority sample according to their distance
            # from u
            to_sort_euc = zip(
                nn_min_euc_ind[random_idx], np.arange(len(X_min)))
            _, sorted_by_euc_ind = zip(*(sorted(to_sort_euc,
                                                key=lambda x: x[0])))
            to_sort_cos = zip(
                nn_min_cos_ind[random_idx], np.arange(len(X_min)))
            _, sorted_by_cos_ind = zip(*(sorted(to_sort_cos,
                                                key=lambda x: x[0])))
            # adding the ranks to get the composite similarity measure (called
            # voting in the paper)
            ranked_min_indices = sorted_by_euc_ind + sorted_by_cos_ind
            # sorting the ranking
            to_sort = zip(ranked_min_indices, np.arange(len(X_min)))
            _, sorted_ranking = zip(*(sorted(to_sort, key=lambda x: x[0])))
            # get the indices of the n_neighbors nearest neighbors according
            # to the composite metrics
            min_indices = sorted_ranking[1:(self.n_neighbors + 1)]

            v = X_maj[self.random_state.choice(nn_maj_ind[random_idx])]
            dif1 = u - v
            uu = u + self.random_state.random_sample()*0.3*dif1
            x = X_min[self.random_state.choice(min_indices[1:])]
            dif2 = uu - x
            w = x + self.random_state.random_sample()*0.5*dif2
            samples.append(w)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
