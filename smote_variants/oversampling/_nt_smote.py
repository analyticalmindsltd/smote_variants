"""
This module implements the NT_SMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['NT_SMOTE']

class NT_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{nt_smote,
                            author={Xu, Y. H. and Li, H. and Le, L. P. and
                                        Tian, X. Y.},
                            booktitle={2014 Seventh International Joint
                                        Conference on Computational Sciences
                                        and Optimization},
                            title={Neighborhood Triangular Synthetic Minority
                                    Over-sampling Technique for Imbalanced
                                    Prediction on Small Samples of Chinese
                                    Tourism and Hospitality Firms},
                            year={2014},
                            volume={},
                            number={},
                            pages={534-538},
                            keywords={financial management;pattern
                                        classification;risk management;sampling
                                        methods;travel industry;Chinese
                                        tourism; hospitality firms;imbalanced
                                        risk prediction;minority class samples;
                                        up-sampling approach;neighborhood
                                        triangular synthetic minority
                                        over-sampling technique;NT-SMOTE;
                                        nearest neighbor idea;triangular area
                                        sampling idea;single classifiers;data
                                        excavation principles;hospitality
                                        industry;missing financial indicators;
                                        financial data filtering;financial risk
                                        prediction;MDA;DT;LSVM;logit;probit;
                                        firm risk prediction;Joints;
                                        Optimization;imbalanced datasets;
                                        NT-SMOTE;neighborhood triangular;
                                        random sampling},
                            doi={10.1109/CSO.2014.104},
                            ISSN={},
                            month={July}}
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_application,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 nn_params=None,
                 ss_params=None,
                 *,
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
        ss_params_default = {'n_dim': 3, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
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
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

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

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # find two nearest minority samples
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=self.n_dim,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        samples = self.sample_simplex(X=X_min,
                                        indices=ind,
                                        n_to_sample=n_to_sample)

        #samples = []
        #while len(samples) < n_to_sample:
        #    # select point randomly
        #    idx = self.random_state.randint(len(X_min))
        #    P_1 = X_min[idx]
        #    # find two closest neighbors
        #    P_2 = X_min[ind[idx][1]]
        #    P_3 = X_min[ind[idx][2]]
        #    # generate random point by sampling the specified triangle
        #    r_1 = self.random_state.random_sample()
        #    r_2 = self.random_state.random_sample()
        #    samples.append((P_3 + r_1 * ((P_1 + r_2 * (P_2 - P_1)) - P_3)))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
