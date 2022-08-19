"""
This module implements the Edge_Det_SMOTE method.
"""

import numpy as np

from ..base import fix_density, coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['Edge_Det_SMOTE']

class Edge_Det_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{Edge_Det_SMOTE,
                            author={Kang, Y. and Won, S.},
                            booktitle={ICCAS 2010},
                            title={Weight decision algorithm for oversampling
                                    technique on class-imbalanced learning},
                            year={2010},
                            volume={},
                            number={},
                            pages={182-186},
                            keywords={edge detection;learning (artificial
                                        intelligence);weight decision
                                        algorithm;oversampling technique;
                                        class-imbalanced learning;class
                                        imbalanced data problem;edge
                                        detection algorithm;spatial space
                                        representation;Classification
                                        algorithms;Image edge detection;
                                        Training;Noise measurement;Glass;
                                        Training data;Machine learning;
                                        Imbalanced learning;Classification;
                                        Weight decision;Oversampling;
                                        Edge detection},
                            doi={10.1109/ICCAS.2010.5669889},
                            ISSN={},
                            month={Oct}}

    Notes:
        * This technique is very loosely specified.
    """

    categories = [OverSamplingSimplex.cat_density_based,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 k=5,
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
            k (int): number of neighbors
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
        nn_params = coalesce(nn_params, {})
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, "k", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k # pylint: disable=invalid-name
        self.nn_params = nn_params
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
                                  'k': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_magnitudes(self, X, y):
        """
        Determine the magnitudes.

        Args:
            X (np.array): the feature vectors
            y (np.array): the targets

        Returns:
            np.array: the magnitudes of minority points
        """
        # organizing class labels according to feature ranking
        magnitudes = np.zeros(X.shape[0])
        for idx in range(X.shape[1]):
            to_sort = zip(X[:, idx], np.arange(len(X)), y)
            _, ind, label = zip(*sorted(to_sort, key=lambda x: x[0]))
            # extracting edge magnitudes in this dimension
            for jdx in range(1, len(ind)-1):
                magnitudes[ind[jdx]] = magnitudes[ind[jdx]] + \
                    (label[jdx-1] - label[jdx+1])**2

        # density estimation
        magnitudes = magnitudes[y == self.min_label]
        magnitudes[magnitudes < 0]= 0
        magnitudes = fix_density(magnitudes)
        magnitudes = np.sqrt(magnitudes)
        magnitudes = magnitudes/np.sum(magnitudes)

        return magnitudes

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

        magnitudes = self.determine_magnitudes(X, y)

        # fitting nearest neighbors models to minority samples
        n_neighbors = min([len(X_min), self.k+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        # do the sampling

        samples = self.sample_simplex(X=X_min,
                                        indices=ind,
                                        n_to_sample=n_to_sample,
                                        base_weights=magnitudes)

        #samples = []
        #for _ in range(n_to_sample):
        #    idx = self.random_state.choice(np.arange(len(X_min)), p=magnitudes)
        #    X_a = X_min[idx]
        #    X_b = X_min[self.random_state.choice(ind[idx][1:])]
        #    samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
