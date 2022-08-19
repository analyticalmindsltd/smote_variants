"""
This module implements the SMOBD method.
"""
import warnings

import numpy as np

from sklearn.cluster import OPTICS

from ..base import fix_density
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['SMOBD']

class SMOBD(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{smobd,
                            author={Cao, Q. and Wang, S.},
                            booktitle={2011 International Conference on
                                        Information Management, Innovation
                                        Management and Industrial
                                        Engineering},
                            title={Applying Over-sampling Technique Based
                                     on Data Density and Cost-sensitive
                                     SVM to Imbalanced Learning},
                            year={2011},
                            volume={2},
                            number={},
                            pages={543-548},
                            keywords={data handling;learning (artificial
                                        intelligence);support vector machines;
                                        oversampling technique application;
                                        data density;cost sensitive SVM;
                                        imbalanced learning;SMOTE algorithm;
                                        data distribution;density information;
                                        Support vector machines;Classification
                                        algorithms;Noise measurement;Arrays;
                                        Noise;Algorithm design and analysis;
                                        Training;imbalanced learning;
                                        cost-sensitive SVM;SMOTE;data density;
                                        SMOBD},
                            doi={10.1109/ICIII.2011.276},
                            ISSN={2155-1456},
                            month={Nov},}
    """

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 eta1=0.5,
                 t=1.8,
                 min_samples=5,
                 nn_params={},
                 max_eps=1.0,
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
            eta1 (float): control parameter of density estimation
            t (float): control parameter of noise filtering
            min_samples (int): minimum samples parameter for OPTICS
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            max_eps (float): maximum environment radius parameter for OPTICS
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_in_range(eta1, 'eta1', [0.0, 1.0])
        self.check_greater_or_equal(t, 't', 0)
        self.check_greater_or_equal(min_samples, 'min_samples', 1)
        self.check_greater_or_equal(max_eps, 'max_eps', 0.0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.eta1 = eta1
        self.t = t # pylint: disable=invalid-name
        self.min_samples = min_samples
        self.nn_params = nn_params
        self.max_eps = max_eps
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
                                  'eta1': [0.1, 0.5, 0.9],
                                  't': [1.5, 2.5],
                                  'min_samples': [5],
                                  'max_eps': [0.1, 0.5, 1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_indices_and_density(self, X_min, nn_params):
        """
        Determine the neighborhood structure and the density

        Args:
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbors parameters

        Returns:
            np.array, np.array: the neighborhood structure and the density
        """
        # running the OPTICS technique based on the sklearn implementation
        min_samples = min([len(X_min)-1, self.min_samples])
        optics = OPTICS(min_samples=min_samples,
                   max_eps=self.max_eps,
                   n_jobs=self.n_jobs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            optics.fit(X_min)

        # noise filtering
        cd_average = np.mean(optics.core_distances_)
        rd_average = np.mean(optics.reachability_)
        noise = np.logical_and(optics.core_distances_ > cd_average*self.t,
                                optics.reachability_ > rd_average*self.t)

        # fitting a nearest neighbor model to be able to find
        # neighbors in radius
        n_neighbors = min([len(X_min), self.min_samples+1])

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        indices = nnmt.kneighbors(X_min, return_distance=False)

        # determining the density
        factor_1 = optics.core_distances_
        factor_2 = np.array([len(x)
                        for x in nnmt.radius_neighbors(X_min,
                                                       radius=self.max_eps,
                                                       return_distance=False)])

        if np.all(factor_1 == np.inf) or np.all(factor_2 == 0):
            raise ValueError("factor_1 or factor_2 is not suitable")

        factor_1[factor_1 == np.inf] = max(factor_1[factor_1 != np.inf])*1.1

        factor_1 = factor_1 / max([max(factor_1), 1e-8])
        factor_2 = factor_2 / max([max(factor_2), 1e-8])

        density = factor_1*self.eta1 + factor_2*(1 - self.eta1)

        # setting the density at noisy samples to zero
        density[noise] = 0.0

        density = fix_density(density)

        return indices, density

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

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        try:
            indices, density = self.determine_indices_and_density(X_min,
                                                                  nn_params)
        except ValueError as valueerror:
            return self.return_copies(X, y, valueerror.args[0])

        base_ind = self.random_state.choice(np.arange(X_min.shape[0]),
                                            n_to_sample,
                                            p=density)
        neigh_ind = self.random_state.choice(np.arange(1, indices.shape[1]),
                                             n_to_sample)
        base_vectors = X_min[base_ind]
        neigh_vectors = X_min[indices[base_ind, neigh_ind]]
        random = self.random_state.random_sample(size=base_vectors.shape)

        samples = base_vectors + (neigh_vectors - base_vectors) *  random

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'eta1': self.eta1,
                't': self.t,
                'min_samples': self.min_samples,
                'nn_params': self.nn_params,
                'max_eps': self.max_eps,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
