"""
This module implements the Safe_Level_SMOTE method.
"""

import numpy as np

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['Safe_Level_SMOTE']

class Safe_Level_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{safe_level_smote,
                        author = {
                            Bunkhumpornpat, Chumphol and Sinapiromsaran,
                        Krung and Lursinsap, Chidchanok},
                        title = {Safe-Level-SMOTE: Safe-Level-Synthetic
                                Minority Over-Sampling TEchnique for
                                Handling the Class Imbalanced Problem},
                        booktitle = {Proceedings of the 13th Pacific-Asia
                                    Conference on Advances in Knowledge
                                    Discovery and Data Mining},
                        series = {PAKDD '09},
                        year = {2009},
                        isbn = {978-3-642-01306-5},
                        location = {Bangkok, Thailand},
                        pages = {475--482},
                        numpages = {8},
                        url = {http://dx.doi.org/10.1007/978-3-642-01307-2_43},
                        doi = {10.1007/978-3-642-01307-2_43},
                        acmid = {1533904},
                        publisher = {Springer-Verlag},
                        address = {Berlin, Heidelberg},
                        keywords = {Class Imbalanced Problem, Over-sampling,
                                    SMOTE, Safe Level},
                    }

    Notes:
        * The original method was not prepared for the case when no minority
            sample has minority neighbors.
    """

    categories = [OverSampling.cat_borderline,
                  OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params= {},
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

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1.0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
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
                                  'n_neighbors': [3, 5, 7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def slp_sln(self, y, indices):
        """
        Creating an array of allowed positive negative pairs
        and the corresponding safe level scores.

        Args:
            y (np.array): the target labels
            indices (np.array): the neighborhood structure

        Returns:
            np.array: an array with the first column containing the positive
                        index, the second the neighbor index, the third and
                        fourth the corresponding safe levels
        """
        minority_indices = np.where(y == self.min_label)[0]

        safe_levels = np.sum(y[indices[:, 1:]] == self.min_label, axis=1)

        pos_slp = minority_indices[safe_levels[minority_indices] > 0]
        null_slp = minority_indices[safe_levels[minority_indices] == 0]

        part0 = np.vstack([np.repeat(pos_slp, indices.shape[1] - 1),
                            indices[pos_slp, 1:].flatten()]).T

        pos_sln = np.where(safe_levels[indices[null_slp, 1:]] > 0)
        part1 = np.vstack([null_slp[pos_sln[0]],
                            indices[null_slp, 1:][pos_sln]]).T

        pairs = np.vstack([part0, part1])
        return np.vstack([pairs.T, safe_levels[pairs[:, 0]], safe_levels[pairs[:, 1]]]).T

    def generate_samples(self, X, y, slp_sln, n_to_sample):
        """
        Generate samples.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            slp_sln (np.array): the slp-sln matrix
            n_to_smaple (int): the number of samples to generate

        Returns:
            np.array, np.array: the extended training set
        """
        if len(slp_sln) == 0:
            return self.return_copies(X, y, "No neighbors passing the "\
                                                            "safe level test")

        base_indices = \
            self.random_state.choice(np.arange(len(slp_sln)), n_to_sample)

        gap = self.random_state.random_sample(size=(n_to_sample, X.shape[1]))

        slp = slp_sln[base_indices][:,2]
        sln = slp_sln[base_indices][:,3]

        gap = (gap.T * np.where(np.logical_and(sln == 0, slp > 0), 0.0, 1.0)).T

        mask = slp > sln
        gap[mask] = (gap[mask].T * sln[mask]/slp[mask]).T

        mask = slp < sln
        gap[mask] = (gap[mask].T * slp[mask]/sln[mask]).T
        gap[mask] = (gap[mask].T + (1.0 - slp[mask]/sln[mask])).T

        samples = X[slp_sln[base_indices][:,0]] + \
                    ((X[slp_sln[base_indices][:,1]] - X[slp_sln[base_indices][:,0]]) * gap)

        return (np.vstack([X, samples]),
               np.hstack([y, np.repeat(self.min_label, len(samples))]))

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

        # fitting nearest neighbors model
        n_neighbors = min([self.n_neighbors+1, len(X)])

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        indices = nnmt.kneighbors(X, return_distance=False)

        slp_sln = self.slp_sln(y, indices)

        return self.generate_samples(X, y, slp_sln, n_to_sample)


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
