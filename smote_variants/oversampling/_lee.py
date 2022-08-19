"""
This module implements the Lee method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['Lee']

class Lee(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @inproceedings{lee,
                             author = {Lee, Jaedong and Kim,
                                 Noo-ri and Lee, Jee-Hyong},
                             title = {An Over-sampling Technique with Rejection
                                        for Imbalanced Class Learning},
                             booktitle = {Proceedings of the 9th International
                                            Conference on Ubiquitous
                                            Information Management and
                                            Communication},
                             series = {IMCOM '15},
                             year = {2015},
                             isbn = {978-1-4503-3377-1},
                             location = {Bali, Indonesia},
                             pages = {102:1--102:6},
                             articleno = {102},
                             numpages = {6},
                             doi = {10.1145/2701126.2701181},
                             acmid = {2701181},
                             publisher = {ACM},
                             address = {New York, NY, USA},
                             keywords = {data distribution, data preprocessing,
                                            imbalanced problem, rejection rule,
                                            synthetic minority oversampling
                                            technique}
                            }
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 rejection_level=0.5,
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
            n_neighbors (int): number of neighbors in nearest neighbor
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            rejection_level (float): the rejection level of generated samples,
                                        if the fraction of majority labels in
                                        the local environment is higher than
                                        this number, the generated point is
                                        rejected
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(rejection_level, "rejection_level", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.rejection_level = rejection_level
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
                                  'n_neighbors': [3, 5, 7],
                                  'rejection_level': [0.3, 0.5, 0.7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def generate_samples(self,
                            *,
                            y,
                            X_min,
                            n_to_sample,
                            ind_min,
                            nnmt):
        """
        Generate samples

        Args:
            y (np.array): all target labels
            X_min (np.array): minority samples
            n_to_sample (int): the number of samples to generate
            ind_min (np.array): the minority neighborhood structure
            nnmt (obj): a fitted nearest neighbor model

        Returns:
            np.array: the generated samples
        """
        samples = np.zeros(shape=(0, X_min.shape[1]), dtype=float)

        rejection_level = self.rejection_level
        tries = 0
        while samples.shape[0] < n_to_sample:
            # generating at least 20 samples always to avoid iterating
            # to generate 1 sample in each iteration
            n_missing = np.max([n_to_sample - samples.shape[0], 20])

            samples_new = self.sample_simplex(X=X_min,
                                                indices=ind_min,
                                                n_to_sample=n_missing)
            ind_new = nnmt.kneighbors(samples_new, return_distance=False)
            maj_frac = np.sum(y[ind_new] == self.maj_label, axis=1)\
                                                        /self.n_neighbors
            if np.sum(maj_frac < rejection_level) == 0:
                tries = tries + 1

            if tries > 5:
                rejection_level = rejection_level + 0.1
                tries = 0

            samples = np.vstack([samples,
                                 samples_new[maj_frac < rejection_level]])

        samples = samples[:n_to_sample]

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
            return self.return_copies(X, y, "Sampling is not neeed")

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors models to find neighbors of minority
        # samples in the total data and in the minority datasets
        n_neighbors = min([len(X_min), self.n_neighbors])
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)

        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nn_min= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
        nn_min.fit(X_min)
        ind_min = nn_min.kneighbors(X_min, return_distance=False)

        # do the sampling, we implemented a continuous tweaking of rejection
        # levels in order to fix situations when no unrejectable data can
        # be can be generated

        samples = self.generate_samples(y=y,
                                        X_min=X_min,
                                        n_to_sample=n_to_sample,
                                        ind_min=ind_min,
                                        nnmt=nnmt)

        #samples = []
        #passed = 0
        #trial = 0
        #rejection_level = self.rejection_level
        #while len(samples) < n_to_sample:
        #    # checking if we managed to generate a single data in 1000 trials
        #    if passed == trial and passed > 1000:
        #        rejection_level = rejection_level + 0.1
        #        trial = 0
        #        passed = 0
        #    trial = trial + 1
        #    # generating random point
        #    idx = self.random_state.randint(len(X_min))
        #    random_neighbor_idx = self.random_state.choice(ind_min[idx][1:])
        #    X_a = X_min[idx]
        #    X_b = X_min[random_neighbor_idx]
        #    random_point = self.sample_between_points(X_a, X_b)
        #    # checking if the local environment is above the rejection level
        #    ind_new = nnmt.kneighbors(random_point.reshape(1, -1), return_distance=False)
        #    maj_frac = np.sum(y[ind_new[0]] == self.maj_label)/self.n_neighbors
        #
        #    if maj_frac < rejection_level:
        #        samples.append(random_point)
        #    else:
        #        passed = passed + 1

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'rejection_level': self.rejection_level,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
