"""
This module implements the MSMOTE method.
"""

import numpy as np

from ..base import coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['MSMOTE']

class MSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{msmote,
                             author = {Hu, Shengguo and Liang,
                                 Yanfeng and Ma, Lintao and He, Ying},
                             title = {MSMOTE: Improving Classification
                                        Performance When Training Data
                                        is Imbalanced},
                             booktitle = {Proceedings of the 2009 Second
                                            International Workshop on
                                            Computer Science and Engineering
                                            - Volume 02},
                             series = {IWCSE '09},
                             year = {2009},
                             isbn = {978-0-7695-3881-5},
                             pages = {13--17},
                             numpages = {5},
                             url = {https://doi.org/10.1109/WCSE.2009.756},
                             doi = {10.1109/WCSE.2009.756},
                             acmid = {1682710},
                             publisher = {IEEE Computer Society},
                             address = {Washington, DC, USA},
                             keywords = {imbalanced data, over-sampling,
                                        SMOTE, AdaBoost, samples groups,
                                        SMOTEBoost},
                            }

    Notes:
        * The original method was not prepared for the case when all
            minority samples are noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
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
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)

        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
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

    def generate_samples(self, *, X_min, X, indices, sample_type, n_to_sample):
        """
        Generate samples.

        Args:
            X_min (np.array): the minority samples
            X (np.array): all training vectors
            indices (np.array): the neighborhood structure
            sample_type (np.array): the sample types
            n_to_sample (int): number of samples to generate

        Returns:
            np.array: the generated samples
        """
        not_noise_mask = sample_type != 'NOI'
        X_not_noise = X_min[not_noise_mask] # pylint: disable=invalid-name
        st_not_noise = sample_type[not_noise_mask]
        ind_not_noise = indices[not_noise_mask]

        base_indices = self.random_state.choice(X_not_noise.shape[0],
                                                n_to_sample)

        neighbor_indices = np.array([1] * n_to_sample)
        neighbor_indices[st_not_noise[base_indices] == 'SEC'] = \
            self.random_state.choice(np.arange(1, indices.shape[1]),
                                     np.sum(st_not_noise[base_indices] == 'SEC'))

        neighbor_indices = ind_not_noise[base_indices,
                                         neighbor_indices]

        base_vectors = X_not_noise[base_indices]
        neighbor_vectors = X[neighbor_indices]

        samples = base_vectors + (neighbor_vectors - base_vectors) \
                       * self.random_state.random_sample(base_vectors.shape)

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
        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        # fitting the nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors+1])

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        indices = nnmt.kneighbors(X_min, return_distance=False)

        n_p = np.sum(y[indices[:, 1:]] == self.min_label, axis=1)
        sample_type = np.array(['BOR'] * n_p.shape[0])
        sample_type[n_p == (n_neighbors - 1)] = 'SEC'
        sample_type[n_p == 0] = 'NOI'

        if np.all(sample_type == 'NOI'):
            return self.return_copies(X, y, "All samples are noise")

        samples = self.generate_samples(X_min=X_min,
                                        X=X,
                                        indices=indices,
                                        sample_type=sample_type,
                                        n_to_sample=n_to_sample)

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
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
