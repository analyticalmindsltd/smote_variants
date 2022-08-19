"""
This module implements the CCR method.
"""

import numpy as np

from sklearn.metrics import pairwise_distances

from ..base import OverSampling
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
        * The number of synthetic samples to generate is hacked for reasonable results
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 energy=1.0,
                 scaling=0.0,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
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
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(energy, "energy", 0)
        self.check_greater_or_equal(scaling, "scaling", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.energy = energy
        self.scaling = scaling
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
                                  'energy': [0.001, 0.0025, 0.005,
                                             0.01, 0.025, 0.05, 0.1,
                                             0.25, 0.5, 1.0, 2.5, 5.0,
                                             10.0, 25.0, 50.0, 100.0],
                                  'scaling': [0.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def taxicab_sample(self, n_dim, radius):
        """
        Generate one taxi cab sample.

        Args:
            n_dim (int): the number of dimensions
            radius (float): the radius to be used

        Returns:
            np.array: one taxicab sample
        """
        sample = []
        random_numbers = self.random_state.rand(n_dim)

        for idx in range(n_dim):
            # spread = r - np.sum(np.abs(sample))
            spread = radius
            if len(sample) > 0:
                spread -= abs(sample[-1])
            sample.append(spread * (2 * random_numbers[idx] - 1))

        return self.random_state.permutation(sample)

    def taxicab_samples(self, n_dim, radius, n_samples=1):
        """
        Generate n taxicab samples.

        Args:
            n_dim (int): the number of dimensions.
            radius (float): the radius to be used
            n_samples (int): the number of samples to be generated

        Returns:
            np.array: the generated samples
        """
        return np.vstack([self.taxicab_sample(n_dim, radius) \
                                        for _ in range(n_samples)])

    def random_majority_offset(self, n_dim):
        """
        Generate random majority offset.

        Args:
            n_dim (int): number of dimensions

        Returns:
            np.array: the random majority offset
        """
        r_num = self.random_state.rand(n_dim)
        r_num = 1e-6 * r_num + 1e-6
        r_sign = self.random_state.choice([-1.0, 1.0], n_dim)

        return r_num * r_sign

    def update_majority_point(self, dist, majority_point, minority_point):
        """
        Updating the majority point to break up ties with the minority point

        Args:
            float (dist): the distance
            majority_point (np.array): a majority point
            minority_pont (np.array): a minority point

        Returns:
            float, np.array: the updated dist and majority_point
        """
        if dist < 1e-20:
            majority_point = majority_point + \
                            self.random_majority_offset(majority_point.shape[0])
            dist = np.sum(np.abs(minority_point - majority_point))
            return dist, majority_point
        return dist, majority_point

    def determine_radii_current_majority(self, majority, distances, idx, sorted_distances):
        """
        Determines the radius for the minority sample idx and also returns the
        majority (current_majority) the algorithm arrives to.

        Args:
            majority (np.array): majority samples
            distances (np.array): the distance matrix
            idx (int): the index of the minority samples to work on
            sorted_distances (np.array): the sorted distances of the minority
                                            item
        """
        last_distance = 0.0
        remaining_energy = self.energy * (majority.shape[1] ** self.scaling)
        radius = 0.0
        current_majority = 0
        for current_majority in range(len(majority)):
            if remaining_energy <= 0:
                remaining_energy = 0
                break

            radius_change = remaining_energy / (current_majority + 1.0)

            dist = distances[idx, sorted_distances[current_majority]]
            if dist >= radius + radius_change:
                radius += radius_change
                break

            curr_maj_idx = sorted_distances[current_majority]
            radius_change = np.max([distances[idx, curr_maj_idx] - last_distance, 0])
            radius += radius_change
            remaining_energy -= radius_change * (current_majority + 1.0)

            last_distance = distances[idx, sorted_distances[current_majority - 1]]


        # if current_majority == len(majority) - 1 then adjust
        multiplier_flag = int(current_majority == (len(majority) - 1))
        radius += (remaining_energy / max(1, current_majority)) * multiplier_flag

        return radius, current_majority

    def determine_radii_translations(self, minority, majority):
        """
        Determine the radii and the translations.

        Args:
            minority (np.array): the minority samples
            majority (np.array): the majority samples

        Returns:
            np.array, np.array: the radii and the translations
        """
        distances = pairwise_distances(minority, majority, metric='l1')

        radii = np.zeros(len(minority))
        translations = np.zeros(majority.shape)

        for idx, minority_point in enumerate(minority):
            sorted_distances = np.argsort(distances[idx])
            radius, current_majority = self.determine_radii_current_majority(majority,
                                                                            distances,
                                                                            idx,
                                                                            sorted_distances)

            radii[idx] = radius

            for jdx in range(current_majority):
                majority_point = majority[sorted_distances[jdx]]
                dist = distances[idx, sorted_distances[jdx]]

                dist, majority_point = self.update_majority_point(dist,
                                                                majority_point,
                                                                minority_point)

                translation = (radius - dist) / dist * (majority_point - minority_point)
                translations[sorted_distances[jdx]] += translation

        return radii, translations

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

        minority = X[y == self.min_label]
        majority = X[y == self.maj_label]

        radii, translations = self.determine_radii_translations(minority,
                                                                majority)

        majority = majority.astype(float)
        majority += translations

        # this is changed to the ratio of sums for more robustness
        radii_inv_sum = len(radii) / np.sum(radii)
        #radii_inv_sum = 1.0 / np.sum(1.0 / radii)

        appended = [np.zeros(shape=(0, majority.shape[1]))]
        for idx, minority_point in enumerate(minority):
            # this determination of synthetic samples is hacked
            # to give reasonable results
            synthetic_samples = n_to_sample / len(minority) * (1.0 / radii[idx]) / radii_inv_sum
            synthetic_samples = int(np.round(synthetic_samples))
            if synthetic_samples > 0:
                tc_samples = self.taxicab_samples(minority_point.shape[0],
                                                    radii[idx],
                                                    synthetic_samples)
                appended.append(minority_point + tc_samples)

        appended = np.vstack(appended)

        return (np.vstack([X, appended]),
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
                **OverSampling.get_params(self)}
