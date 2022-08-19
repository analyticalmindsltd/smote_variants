"""
This module implements the SMMO method.
"""
import warnings

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from ..base import instantiate_obj

from .._logger import logger
_logger= logger

__all__= ['SMMO']

class SMMO(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{smmo,
                            author = {de la Calleja, Jorge and Fuentes, Olac
                                        and González, Jesús},
                            booktitle= {Proceedings of the Twenty-First
                                        International Florida Artificial
                                        Intelligence Research Society
                                        Conference},
                            year = {2008},
                            month = {01},
                            pages = {276-281},
                            title = {Selecting Minority Examples from
                                    Misclassified Data for Over-Sampling.}
                            }

    Notes:
        * In this paper the ensemble is not specified. I have selected
            some very fast, basic classifiers.
        * Also, it is not clear what the authors mean by "weighted distance".
        * The original technique is not prepared for the case when no minority
            samples are classified correctly be the ensemble.
    """

    categories = [OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_classifier,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 ensemble=[('sklearn.discriminant_analysis',
                            'QuadraticDiscriminantAnalysis',
                            {}),
                            ('sklearn.tree',
                            'DecisionTreeClassifier',
                            {'random_state': 2}),
                            ('sklearn.naive_bayes',
                            'GaussianNB',
                            {})],
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
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            ensemble (list): list of classifiers, if None, default list of
                                classifiers is used
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.ensemble = ensemble
        self.ensemble_objs = [instantiate_obj(e) for e in ensemble]
        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        ensembles = [[('sklearn.discriminant_analysis',
                        'QuadraticDiscriminantAnalysis',
                        {}),
                        ('sklearn.tree',
                        'DecisionTreeClassifier',
                        {'random_state': 2}),
                        ('sklearn.naive_bayes',
                        'GaussianNB',
                        {})]]
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'ensemble': ensembles}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def generate_samples(self, X_min, ind, n_to_sample):
        """
        Generate the samples.

        Args:
            X_min (np.array): minority samples
            ind (np.array): neighborhood structure
            n_to_sample (int): number of samples to generate

        Returns:
            np.array: the generated samples
        """
        X_means = np.mean(X_min[ind[:, 1:]], axis=1) # pylint: disable=invalid-name
        base_indices = self.random_state.choice(np.arange(X_means.shape[0]),
                                                n_to_sample)
        base_ind, base_counts = np.unique(base_indices, return_counts=True)
        samples = []
        for idx, index in enumerate(base_ind):
            cluster = np.array([X_means[index]])
            indices = np.array([np.hstack([np.array([0]), np.arange(len(cluster))])])
            X_vertices = X_min[ind[index][indices[0]][1:]]
            samples.append(self.sample_simplex(X=cluster,
                                            indices=indices,
                                            n_to_sample=base_counts[idx],
                                            X_vertices=X_vertices))

        samples = np.vstack(samples)
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

        # training and in-sample prediction (out-of-sample by k-fold cross
        # validation might be better)
        predictions = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for ens in self.ensemble_objs:
                predictions.append(ens.fit(X, y).predict(X))

        # constructing ensemble prediction
        pred = np.where(np.mean(np.vstack(predictions), axis=0) > 0.5,
                        1, 0)

        # create mask of minority samples to sample
        mask_to_sample = np.where(np.logical_and(pred != y,
                                                 y == self.min_label))[0]
        if len(mask_to_sample) < 2:
            return self.return_copies(X, y, "Not enough minority samples "\
                            f"selected {len(mask_to_sample)}")

        X_min = X[y == self.min_label]
        X_min_to_sample = X[mask_to_sample] # pylint: disable=invalid-name

        # fitting nearest neighbors model for sampling
        n_neighbors = min([len(X_min), self.n_neighbors + 1])

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min_to_sample, return_distance=False)

        samples = self.generate_samples(X_min, ind, n_to_sample)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'ensemble': self.ensemble,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self, deep)}
