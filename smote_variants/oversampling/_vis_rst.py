"""
This module implements the VIS_RST method.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['VIS_RST']

class VIS_RST(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{vis_rst,
                            author="Borowska, Katarzyna
                            and Stepaniuk, Jaroslaw",
                            editor="Saeed, Khalid
                            and Homenda, Wladyslaw",
                            title="Imbalanced Data Classification: A Novel
                                    Re-sampling Approach Combining Versatile
                                    Improved SMOTE and Rough Sets",
                            booktitle="Computer Information Systems and
                                        Industrial Management",
                            year="2016",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="31--42",
                            isbn="978-3-319-45378-1"
                            }

    Notes:
        * Replication of DANGER samples will be removed by the last step of
            noise filtering.
        * The rules in the paper do not cover all cases
    """

    categories = [OverSamplingSimplex.cat_changes_majority,
                  OverSamplingSimplex.cat_noise_removal,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
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
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
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
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def relabel_boundary(self, X, y, nn_params):
        """
        Relabel the boundary vectors

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array: the new minority samples and the updated
                                target labels
        """
        # fitting nearest neighbors model to determine boundary region
        n_neighbors = min([len(X), self.n_neighbors + 1])

        X_maj = X[y == self.maj_label]

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        ind = nnmt.kneighbors(X_maj, return_distance=False)

        # determining boundary region of majority samples
        boundary = np.sum(y[ind] == self.maj_label, axis=1) != n_neighbors

        y_maj = y[y == self.maj_label]
        y_maj[boundary] = self.min_label
        y[y == self.maj_label] = y_maj

        # extracting new minority and majority set
        X_min = X[y == self.min_label]

        return X_min, y

    def label_minority_samples(self, X, y, X_min, nn_params):
        """
        Label the minority samples.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            X_min (np.array): the minority vectors
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the labels
        """
        # fitting nearest neighbors model to determine boundary region
        n_neighbors = min([len(X), self.n_neighbors + 1])

        # labeling minority samples
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        indices = nnmt.kneighbors(X_min, return_distance=False)

        # extracting labels
        labels = np.array(['DAN'] * X_min.shape[0])

        min_class_neighbors = \
                    np.sum(y[indices[:, 1:]] == self.maj_label, axis=1)

        labels[min_class_neighbors == (n_neighbors - 1)] = 'NOI'
        labels[min_class_neighbors < (n_neighbors - 1) / 2] = 'SAF'

        return labels

    def set_mode(self, X_min, labels):
        """
        Determine the mode of oversampling.

        Args:
            X_min (np.array): the minority samples
            labels (np.array): the labels

        Returns:
            str: the mode
        """
        # extracting the number of different labels (noise is not used)
        safe = np.sum(labels == 'SAF')
        danger = np.sum(labels == 'DAN')

        if safe == 0:
            mode = 'no_safe'
        elif danger > 0.3*len(X_min):
            mode = 'high_complexity'
        else:
            mode = 'low_complexity'

        _logger.info("%s: selected mode: %s", self.__class__.__name__, mode)

        return mode

    def sampling_high_complexity(self, X_min, labels, ind_min, n_to_sample):
        """
        Implementation of the sampling rules for the high complexity case.

        Args:
            X_min (np.array): the minority samples
            labels (np.array): the minority labels
            ind_min (np.array): the neighborhood structure
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        # do the sampling
        not_noise_mask = labels != 'NOI'
        X_min_not_noise = X_min[not_noise_mask] # pylint: disable=invalid-name
        danger_mask = labels[not_noise_mask] == 'DAN'
        ind_not_noise = ind_min[not_noise_mask]

        samples = np.zeros((0, X_min.shape[1]))

        while samples.shape[0] < n_to_sample:
            n_missing = np.max([n_to_sample - samples.shape[0], 10])
            base_indices = self.random_state.choice(X_min_not_noise.shape[0],
                                                    n_missing)
            base_danger = base_indices[danger_mask[base_indices]]
            samples = np.vstack([samples, X_min_not_noise[base_danger]])

            base_not_danger = base_indices[~danger_mask[base_indices]]

            n_missing = base_not_danger.shape[0]

            base_not_danger = ~danger_mask

            samples_tmp = self.sample_simplex(X=X_min_not_noise[base_not_danger],
                                          indices=ind_not_noise[base_not_danger, 1:],
                                          n_to_sample=n_missing,
                                          X_vertices=X_min)

            samples = np.vstack([samples, samples_tmp])

        samples = samples[:n_to_sample]

        return samples

    def sampling_low_complexity(self, X_min, labels, ind_min, n_to_sample):
        """
        Implementation of the sampling rules for the high complexity case.

        Args:
            X_min (np.array): the minority samples
            labels (np.array): the minority labels
            ind_min (np.array): the neighborhood structure
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """

        # do the sampling
        not_noise_mask = labels != 'NOI'
        X_min_not_noise = X_min[not_noise_mask] # pylint: disable=invalid-name
        danger_mask = labels[not_noise_mask] == 'DAN'
        ind_not_noise = ind_min[not_noise_mask]

        samples = np.zeros((0, X_min.shape[1]))

        while samples.shape[0] < n_to_sample:
            n_missing = np.max([n_to_sample - samples.shape[0], 10])
            base_indices = self.random_state.choice(X_min_not_noise.shape[0],
                                                    n_missing)
            base_not_danger = base_indices[~danger_mask[base_indices]]
            samples = np.vstack([samples, X_min_not_noise[base_not_danger]])

            base_danger = base_indices[danger_mask[base_indices]]

            n_missing = base_danger.shape[0]

            base_danger = danger_mask

            samples_tmp = self.sample_simplex(X=X_min_not_noise[base_danger],
                                          indices=ind_not_noise[base_danger, 1:],
                                          n_to_sample=n_missing,
                                          X_vertices=X_min)

            samples = np.vstack([samples, samples_tmp])

        samples = samples[:n_to_sample]

        return samples

    def sampling_otherwise(self, X_min, ind_min, n_to_sample):
        """
        Implementation of the sampling rules for the high complexity case.

        Args:
            X_min (np.array): the minority samples
            ind_min (np.array): the neighborhood structure
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """

        samples = self.sample_simplex(X=X_min,
                                        indices=ind_min,
                                        n_to_sample=n_to_sample)

        return samples

    def generate_samples(self, *, X_min, nn_params, n_to_sample, labels, mode):
        """
        Generate samples.

        Args:
            X_min (np.array): all minority vectors
            nn_params (dict): the nearest neighbor parameters
            n_to_sample (int): the number of samples to generate
            labels (np.array): the labels of the minority samples
            mode (str): the sampling mode

        Returns:
            np.array: the generated samples
        """
        # fitting nearest neighbors to find the neighbors of minority elements
        # among minority elements
        n_neighbors_min = min([len(X_min), self.n_neighbors + 1])
        nn_min = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors_min,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
        nn_min.fit(X_min)
        ind_min = nn_min.kneighbors(X_min, return_distance=False)

        # implementation of sampling rules depending on the mode
        # the rules are not covering all cases

        if mode == 'high_complexity':
            samples = self.sampling_high_complexity(X_min,
                                                    labels,
                                                    ind_min,
                                                    n_to_sample)
        elif mode == 'low_complexity':
            samples = self.sampling_low_complexity(X_min,
                                                    labels,
                                                    ind_min,
                                                    n_to_sample)
        else:
            samples = self.sampling_otherwise(X_min,
                                                ind_min,
                                                n_to_sample)

        return samples

    def noise_removal(self, X, y, samples, nn_params):
        """
        Remove the noise from the genrerated samples.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            samples (np.array): the generated samples
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array: the updated samples
        """
        n_neighbors = min([len(X), self.n_neighbors + 1])

        # final noise removal by removing those minority samples generated
        # and not belonging to the lower approximation
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        ind_check = nnmt.kneighbors(samples, return_distance=False)

        num_maj_mask = \
            np.sum(y[ind_check[:, 1:]] == self.maj_label, axis=1) == 0

        return samples[num_maj_mask]

    def check_all_boundary(self, y):
        """
        Check if too many boundary samples identified.

        Args:
            y (np.array): the target labels
        """
        if np.sum(y == self.maj_label) <= np.sum(y == self.min_label):
            raise ValueError("too many majority samples identified as boundary")

    def check_labels(self, labels):
        """
        Check if not all minority samples are noise

        Args:
            labels (np.array): the minority labels
        """
        if len(np.unique(labels)) == 1 and labels[0] == 'NOI':
            raise ValueError("all minority samples identified as noise")

    def check_all_noise(self, y):
        """
        Check if there is only one class left after noise removal

        Args:
             y (np.array): the target labels
        """
        if len(np.unique(y)) == 1:
            raise ValueError("one class removed as noise")

    def all_steps(self, X, y, nn_params, n_to_sample):
        """
        Execute all steps of the algorithm.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            nn_params (dict): nearest neighbor parameters
            n_to_sample (int): the number of samples to generate
        """
        X_min, y = self.relabel_boundary(X, y, nn_params)

        self.check_all_boundary(y)

        labels = self.label_minority_samples(X, y, X_min, nn_params)

        self.check_labels(labels)

        mode = self.set_mode(X_min, labels)

        samples = self.generate_samples(X_min=X_min,
                                        nn_params=nn_params,
                                        n_to_sample=n_to_sample,
                                        labels=labels,
                                        mode=mode)

        samples = self.noise_removal(X, y, samples, nn_params)

        self.check_all_noise(y)

        return samples

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

        # standardizing the data
        X_orig, y_orig = X, y

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = y.copy()

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        try:
            samples = self.all_steps(X, y, nn_params, n_to_sample)
        except ValueError as value_error:
            return self.return_copies(X_orig, y_orig, value_error.args[0])

        return (scaler.inverse_transform(np.vstack([X, samples])),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
