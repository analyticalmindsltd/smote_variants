"""
This module implements the SMOTE_OUT method.
"""

import numpy as np

from ..base import coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling

from .._logger import logger
_logger= logger

__all__= ['SMOTE_OUT']

class SMOTE_OUT(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                      title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An
                                enhancement strategy to handle imbalance in
                                data level},
                      author={Fajri Koto},
                      journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                      year={2014},
                      pages={280-284}
                    }
    """

    categories = [OverSampling.cat_extensive,
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
            n_neighbors (int): parameter of the NearestNeighbors component
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

    def generate_samples(self, *, X, X_maj, X_min, minority_indices,
                            n_to_sample, maj_indices, min_indices):
        """
        Generate samples

        Args:
            X (np.array): all training vectors
            X_maj (np.array): majority vectors
            X_min (np.array): minority vectors
            minority_indices (np.array): the minority indices
            n_to_sample (int): number of samples to generate
            maj_indices (np.array): majority neighborhood structure
            min_indices (np.array): minority neighborhood structure

        Returns:
            np.array: the generated samples
        """
        base_ind = self.random_state.choice(np.arange(len(minority_indices)),
                                            n_to_sample)

        u = X[minority_indices[base_ind]] # pylint: disable=invalid-name
        neigh_ind = self.random_state.choice(np.arange(0, maj_indices.shape[1]),
                                             n_to_sample)
        v = X_maj[maj_indices[base_ind, neigh_ind]] # pylint: disable=invalid-name
        uu = u + self.random_state.random_sample(u.shape) * 0.3 * (u - v) # pylint: disable=invalid-name
        min_neigh_ind = self.random_state.choice(np.arange(1, min_indices.shape[1]),
                                                 n_to_sample)
        x = X_min[min_indices[base_ind, min_neigh_ind]] # pylint: disable=invalid-name
        return x + self.random_state.random_sample(x.shape) * 0.5 * (uu - x)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        # nearest neighbors among minority points
        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        n_neighbors_min = min([len(X_min), self.n_neighbors+1])
        nn_min= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors_min,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
        nn_min.fit(X_min)

        min_indices = nn_min.kneighbors(X_min, return_distance=False)
        # nearest neighbors among majority points
        n_neighbors_maj = min([len(X_maj), self.n_neighbors+1])
        nn_maj= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors_maj,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
        nn_maj.fit(X_maj)
        maj_indices = nn_maj.kneighbors(X_min, return_distance=False)

        samples = self.generate_samples(X=X, X_maj=X_maj, X_min=X_min,
                    minority_indices=minority_indices, n_to_sample=n_to_sample,
                    maj_indices=maj_indices, min_indices=min_indices)


        # generate samples
        #samples = []
        #for _ in range(n_to_sample):
        #    # implementation of Algorithm 1 in the paper
        #    random_idx = self.random_state.choice(np.arange(len(minority_indices)))
        #    u = X[minority_indices[random_idx]]
        #    v = X_maj[self.random_state.choice(maj_indices[random_idx])]
        #    dif1 = u - v
        #    uu = u + self.random_state.random_sample()*0.3*dif1
        #    x = X_min[self.random_state.choice(min_indices[random_idx][1:])]
        #    dif2 = uu - x
        #    w = x + self.random_state.random_sample()*0.5*dif2
        #
        #    samples.append(w)

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
