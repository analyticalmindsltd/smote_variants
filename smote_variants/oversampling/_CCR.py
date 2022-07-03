import numpy as np

from sklearn.metrics import pairwise_distances

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['CCR']

class CCR(OverSampling):
    """
    References:
        * BibTex::

            @article{ccr,
                    author = {Koziarski, Michał and Wozniak, Michal},
                    year = {2017},
                    month = {12},
                    pages = {727–736},
                    title = {CCR: A combined cleaning and resampling algorithm
                                for imbalanced data classification},
                    volume = {27},
                    journal = {International Journal of Applied Mathematics
                                and Computer Science}
                    }

    Notes:
        * Adapted from https://github.com/michalkoziarski/CCR
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 energy=1.0,
                 scaling=0.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            energy (float): energy parameter
            scaling (float): scaling factor
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(energy, "energy", 0)
        self.check_greater_or_equal(scaling, "scaling", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.energy = energy
        self.scaling = scaling
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
                                  'energy': [0.001, 0.0025, 0.005,
                                             0.01, 0.025, 0.05, 0.1,
                                             0.25, 0.5, 1.0, 2.5, 5.0,
                                             10.0, 25.0, 50.0, 100.0],
                                  'scaling': [0.0]}
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        def taxicab_sample(n, r):
            sample = []
            random_numbers = self.random_state.rand(n)

            for i in range(n):
                # spread = r - np.sum(np.abs(sample))
                spread = r
                if len(sample) > 0:
                    spread -= abs(sample[-1])
                sample.append(spread * (2 * random_numbers[i] - 1))

            return self.random_state.permutation(sample)

        minority = X[y == self.min_label]
        majority = X[y == self.maj_label]

        energy = self.energy * (X.shape[1] ** self.scaling)

        distances = pairwise_distances(minority, majority, metric='l1')

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for i in range(len(minority)):
            minority_point = minority[i]
            remaining_energy = energy
            r = 0.0
            sorted_distances = np.argsort(distances[i])
            current_majority = 0

            while True:
                if current_majority > len(majority):
                    break

                if current_majority == len(majority):
                    if current_majority == 0:
                        radius_change = remaining_energy / \
                            (current_majority + 1.0)
                    else:
                        radius_change = remaining_energy / current_majority

                    r += radius_change
                    break

                radius_change = remaining_energy / (current_majority + 1.0)

                dist = distances[i, sorted_distances[current_majority]]
                if dist >= r + radius_change:
                    r += radius_change
                    break
                else:
                    if current_majority == 0:
                        last_distance = 0.0
                    else:
                        cm1 = current_majority - 1
                        last_distance = distances[i, sorted_distances[cm1]]

                    curr_maj_idx = sorted_distances[current_majority]
                    radius_change = distances[i, curr_maj_idx] - last_distance
                    r += radius_change
                    decrease = radius_change * (current_majority + 1.0)
                    remaining_energy -= decrease
                    current_majority += 1

            radii[i] = r

            for j in range(current_majority):
                majority_point = majority[sorted_distances[j]].astype(float)
                d = distances[i, sorted_distances[j]]

                if d < 1e-20:
                    n_maj_point = len(majority_point)
                    r_num = self.random_state.rand(n_maj_point)
                    r_num = 1e-6 * r_num + 1e-6
                    r_sign = self.random_state.choice([-1.0, 1.0], n_maj_point)
                    majority_point += r_num * r_sign
                    d = np.sum(np.abs(minority_point - majority_point))

                translation = (r - d) / d * (majority_point - minority_point)
                translations[sorted_distances[j]] += translation

        majority = majority.astype(float)
        majority += translations

        appended = []
        for i in range(len(minority)):
            minority_point = minority[i]
            synthetic_samples = n_to_sample / (radii[i] * np.sum(1.0 / radii))
            synthetic_samples = int(np.round(synthetic_samples))
            r = radii[i]

            for _ in range(synthetic_samples):
                appended.append(minority_point +
                                taxicab_sample(len(minority_point), r))

        if len(appended) == 0:
            _logger.info("No samples were added")
            return X.copy(), y.copy()

        return (np.vstack([X, np.vstack(appended)]),
                np.hstack([y, np.repeat(self.min_label, len(appended))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'energy': self.energy,
                'scaling': self.scaling,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
