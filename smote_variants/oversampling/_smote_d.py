"""
This module implements the SMOTE_D method.
"""

import numpy as np

from ..base import fix_density
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE_D']

class SMOTE_D(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{smote_d,
                            author="Torres, Fredy Rodr{\'i}guez
                            and Carrasco-Ochoa, Jes{\'u}s A.
                            and Mart{\'i}nez-Trinidad, Jos{\'e} Fco.",
                            editor="Mart{\'i}nez-Trinidad, Jos{\'e} Francisco
                            and Carrasco-Ochoa, Jes{\'u}s Ariel
                            and Ayala Ramirez, Victor
                            and Olvera-L{\'o}pez, Jos{\'e} Arturo
                            and Jiang, Xiaoyi",
                            title="SMOTE-D a Deterministic Version of SMOTE",
                            booktitle="Pattern Recognition",
                            year="2016",
                            publisher="Springer International Publishing",
                            address="Cham",
                            pages="177--188",
                            isbn="978-3-319-39393-3"
                            }

    Notes:
        * Copying happens if two points are the neighbors of each other.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 k=3,
                 *,
                 nn_params={},
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
            k (int): number of neighbors in nearest neighbors component
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

    def generate_samples(self, X_min, ind, counts_ij):
        """
        Generate samples.

        Args:
            X_min (np.array): minority sampes
            ind (np.array): neighborhood structure
            counts_ij (np.array): the counts (density)

        Returns:
            np.array: the generated samples
        """
        # do the sampling
        samples = [np.zeros(shape=(0, X_min.shape[1]))]
        for idx, _ in enumerate(X_min):
            for jdx in range(ind.shape[1]-1):
                while counts_ij[idx][jdx] > 0:
                    if self.random_state.random_sample() < counts_ij[idx][jdx]:
                        translation = X_min[ind[idx][jdx + 1]] - X_min[idx]
                        weight = counts_ij[idx][jdx] + 1
                        samples.append(X_min[idx] + translation/weight)
                    counts_ij[idx][jdx] = counts_ij[idx][jdx] - 1

        return np.vstack(samples)

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

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.k+1])

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        dist, ind = nnmt.kneighbors(X_min)

        # extracting standard deviations of distances
        stds = np.std(dist[:, 1:], axis=1)

        # estimating sampling density
        p_i = fix_density(stds)

        # the other component of sampling density
        p_ij = np.array([fix_density(dist[idx, 1:])
                            for idx in range(dist.shape[0])])
        #p_ij = dist[:, 1:]/np.sum(dist[:, 1:], axis=1)[:, None]

        # number of samples to generate between minority points
        counts_ij = n_to_sample * p_i[:, None] * p_ij

        samples = self.generate_samples(X_min, ind, counts_ij)

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
                **OverSampling.get_params(self)}
