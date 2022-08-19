"""
This module implements the NDO_sampling method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['NDO_sampling']

class NDO_sampling(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ndo_sampling,
                            author={Zhang, L. and Wang, W.},
                            booktitle={2011 International Conference of
                                        Information Technology, Computer
                                        Engineering and Management Sciences},
                            title={A Re-sampling Method for Class Imbalance
                                    Learning with Credit Data},
                            year={2011},
                            volume={1},
                            number={},
                            pages={393-397},
                            keywords={data handling;sampling methods;
                                        resampling method;class imbalance
                                        learning;credit rating;imbalance
                                        problem;synthetic minority
                                        over-sampling technique;sample
                                        distribution;synthetic samples;
                                        credit data set;Training;
                                        Measurement;Support vector machines;
                                        Logistics;Testing;Noise;Classification
                                        algorithms;class imbalance;credit
                                        rating;SMOTE;sample distribution},
                            doi={10.1109/ICM.2011.34},
                            ISSN={},
                            month={Sept}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_application,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 T=0.5,
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
            T (float): threshold parameter
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(T, "T", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = ss_params
        self.T = T # pylint: disable=invalid-name
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
                                  'T': [0.5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_alpha(self, y, ind, dist):
        """
        Determine the alpha value.

        Args:
            y (np.array): the target labels
            ind (np.array): the neighborhood structure
            dist (np.array): the distances

        Returns:
            float: the alpha value
        """
        # calculating the distances between samples in the same and different
        # classes
        indices = np.where(y[ind[:, 1:]] == self.min_label)
        if len(indices[0]) == 0:
            return 0.0
        d_intra_mean = np.mean(dist[:, 1:][indices])
        indices = np.where(y[ind[:, 1:]] == self.maj_label)
        if len(indices[0]) == 0:
            return np.inf
        d_exter_mean = np.mean(dist[:, 1:][indices])
        if d_exter_mean == 0.0:
            return np.inf

        #d_intra = []
        #d_exter = []
        #for i in range(len(X_min)):
        #    min_mask = np.where(y[ind[i][1:]] == self.min_label)[0]
        #    maj_mask = np.where(y[ind[i][1:]] == self.maj_label)[0]
        #    if len(min_mask) > 0:
        #        d_intra.append(np.mean(dist[i][1:][min_mask]))
        #    if len(maj_mask) > 0:
        #        d_exter.append(np.mean(dist[i][1:][maj_mask]))
        #d_intra_mean = np.mean(np.array(d_intra))
        #d_exter_mean = np.mean(np.array(d_exter))

        # calculating the alpha value
        alpha = d_intra_mean/d_exter_mean

        return alpha

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

        # fitting nearest neighbors model to find the neighbors of minority
        # samples among all elements
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        n_neighbors = min([len(X), self.n_neighbors+1])
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        dist, ind = nnmt.kneighbors(X_min)

        alpha = self.determine_alpha(y, ind, dist)

        # deciding if SMOTE is enough
        if alpha < self.T:
            smote = SMOTE(self.proportion,
                          nn_params=nn_params,
                          ss_params=self.ss_params,
                          random_state=self._random_state_init)
            return smote.sample(X, y)

        # do the sampling
        # this is a conditional sampling, which cannot be vectorized
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.randint(len(X_min))
            random_idx = self.random_state.choice(ind[idx][1:])
            # create sample close to the initial minority point
            samples.append(X_min[idx] + (X[random_idx] - X_min[idx])
                           * self.random_state.random_sample()/2.0)
            if y[random_idx] == self.min_label:
                # create another sample close to the neighboring minority point
                samples.append(X[random_idx] + (X_min[idx] - X[random_idx])
                               * self.random_state.random_sample()/2.0)

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
                'ss_params': self.ss_params,
                'T': self.T,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
