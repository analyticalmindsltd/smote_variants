"""
This module implements the SDSMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict, fix_density
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['SDSMOTE']

class SDSMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{sdsmote,
                            author={Li, K. and Zhang, W. and Lu, Q. and
                                        Fang, X.},
                            booktitle={2014 International Conference on
                                        Identification, Information and
                                        Knowledge in the Internet of
                                        Things},
                            title={An Improved SMOTE Imbalanced Data
                                    Classification Method Based on Support
                                    Degree},
                            year={2014},
                            volume={},
                            number={},
                            pages={34-38},
                            keywords={data mining;pattern classification;
                                        sampling methods;improved SMOTE
                                        imbalanced data classification
                                        method;support degree;data mining;
                                        class distribution;imbalanced
                                        data-set classification;over sampling
                                        method;minority class sample
                                        generation;minority class sample
                                        selection;minority class boundary
                                        sample identification;Classification
                                        algorithms;Training;Bagging;Computers;
                                        Testing;Algorithm design and analysis;
                                        Data mining;Imbalanced data-sets;
                                        Classification;Boundary sample;Support
                                        degree;SMOTE},
                            doi={10.1109/IIKI.2014.14},
                            ISSN={},
                            month={Oct}}
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_borderline,
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
            n_neighbors (int): number of neighbors in nearest neighbors
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling params
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
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

    def calculate_density(self, X_min, X_maj, nn_params):
        """
        Calculate the density

        Args:
            X_min (np.array): minority samples
            X_maj (np.array): majority samples
            nn_params (dict): nearest neighbor parameters

        Returns:
            np.array: the density
        """
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=len(X_maj),
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_maj)
        dist, _ = nnmt.kneighbors(X_min)

        # calculating the sum according to S3 in the paper
        S_i = np.sum(dist, axis=1) # pylint: disable=invalid-name

        # calculating average distance according to S5
        S = np.sum(S_i) # pylint: disable=invalid-name
        S_ave = S/(len(X_min)*len(X_maj)) # pylint: disable=invalid-name

        k = np.array([len(nnmt.radius_neighbors(X_min[i].reshape(1, -1), # pylint: disable=invalid-name
                                                S_ave,
                                                return_distance=False))
                                        for i in range(len(X_min))])

        density = k/np.sum(k)

        return fix_density(density)


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

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # fitting nearest neighbors model to find closest majority points to
        # minority samples
        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        density = self.calculate_density(X_min, X_maj, nn_params)

        # fitting nearest neighbors model to minority samples to run
        # SMOTE-like sampling
        n_neighbors = min([len(X_min), self.n_neighbors+1])
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        samples = self.sample_simplex(X=X_min,
                                        indices=ind,
                                        n_to_sample=n_to_sample,
                                        base_weights=density)

        # do the sampling
        #samples = []
        #while len(samples) < n_to_sample:
        #    idx = self.random_state.choice(np.arange(len(density)), p=density)
        #    random_neighbor_idx = self.random_state.choice(ind[idx][1:])
        #    X_a = X_min[idx]
        #    X_b = X_min[random_neighbor_idx]
        #    samples.append(self.sample_between_points(X_a, X_b))

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
                **OverSamplingSimplex.get_params(self)}
