"""
This module implements the ADOMS method.
"""

import numpy as np

from sklearn.decomposition import PCA

from ..base import NearestNeighborsWithMetricTensor, coalesce
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['ADOMS']

class ADOMS(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{adoms,
                            author={Tang, S. and Chen, S.},
                            booktitle={2008 International Conference on
                                        Information Technology and
                                        Applications in Biomedicine},
                            title={The generation mechanism of synthetic
                                    minority class examples},
                            year={2008},
                            volume={},
                            number={},
                            pages={444-447},
                            keywords={medical image processing;
                                        generation mechanism;synthetic
                                        minority class examples;class
                                        imbalance problem;medical image
                                        analysis;oversampling algorithm;
                                        Principal component analysis;
                                        Biomedical imaging;Medical
                                        diagnostic imaging;Information
                                        technology;Biomedical engineering;
                                        Noise generators;Concrete;Nearest
                                        neighbor searches;Data analysis;
                                        Image analysis},
                            doi={10.1109/ITAB.2008.4570642},
                            ISSN={2168-2194},
                            month={May}}
    """

    categories = [OverSampling.cat_dim_reduction,
                  OverSampling.cat_extensive,
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
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): parameter of the nearest neighbor component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state, checks=None)

        self.check_greater_or_equal(proportion, 'proportion', 0.0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @classmethod
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

    def generate_sample_in_neighborhood(self, sample, neighbors):
        """
        Generate artificial sample in a neighborhood.

        Args:
            sample (np.array): one sample point
            neighbors (np.array): the sample's neighbors

        Returns:
            np.array: the generated sample
        """
        # fitting the PCA
        pca = PCA(n_components=1)
        pca.fit(neighbors)

        # extracting the principal direction
        principal_direction = pca.components_[0]

        # do the sampling according to the description in the paper
        random_index = self.random_state.randint(1, len(neighbors))
        random_neighbor = neighbors[random_index]
        diff = np.linalg.norm(random_neighbor - sample)
        rand = self.random_state.random_sample()
        inner_product = np.dot(random_neighbor - sample,
                                principal_direction)
        sign = 1.0 if inner_product > 0.0 else -1.0

        return sample + sign*rand*diff*principal_direction

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
            return self.return_copies(X, y, "no need for sampling")

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = np.min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nearestn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        **(nn_params))
        nearestn.fit(X_min)
        indices = nearestn.kneighbors(X_min, return_distance=False)

        samples = []
        for _ in range(n_to_sample):
            index = self.random_state.randint(len(X_min))
            neighbors = X_min[indices[index]]
            sample = X_min[index]

            samples.append(self.generate_sample_in_neighborhood(sample,
                                                                neighbors))

        return (np.vstack([X, np.vstack(samples)]),
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
