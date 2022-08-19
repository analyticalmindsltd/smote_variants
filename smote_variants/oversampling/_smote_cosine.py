"""
This module implements the SMOTE_Cosine method.
"""

import numpy as np

from sklearn.neighbors import NearestNeighbors

from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SMOTE_Cosine']

class SMOTE_Cosine(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_out_smote_cosine_selected_smote,
                      title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE:
                                An enhancement strategy to handle imbalance
                                in data level},
                      author={Fajri Koto},
                      journal={2014 International Conference on Advanced
                                Computer Science and Information System},
                      year={2014},
                      pages={280-284}
                    }
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
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
            n_neighbors (int): parameter of the NearestNeighbors component
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

    def sampling_formula(self, u, v, x): # pylint: disable=invalid-name
        """
        The sampling formula.

        Args:
            u (np.array): the u vector
            v (np.array): the v vector
            x (np.array): the x vector

        Returns:
            np.array: the generated samples
        """
        uu = u + self.random_state.random_sample(u.shape) * 0.3 * (u - v) # pylint: disable=invalid-name

        return x + self.random_state.random_sample(x.shape) * 0.5 * (uu - x)

    def generate_samples(self, *, X, y, X_maj, X_min,
                        n_to_sample, nn_maj_ind, composite_ind):
        """
        Generate samples

        Args:
            X (np.array): all training vectors
            y (np.array): the target labels
            X_maj (np.array): majority vectors
            X_min (np.array): minority vectors
            n_to_sample (int): number of samples to generate
            nn_maj_ind (np.array): majority neighborhood structure
            composite_ind (np.array): composite neighborhood structure

        Returns:
            np.array: the generated samples
        """
        minority_indices = np.where(y == self.min_label)[0]

        base_ind = self.random_state.choice(np.arange(len(minority_indices)),
                                            n_to_sample)

        neigh_ind = self.random_state.choice(np.arange(0, nn_maj_ind.shape[1]),
                                             n_to_sample)

        min_neigh_ind = self.random_state.choice(np.arange(1, composite_ind.shape[1]),
                                            n_to_sample)

        return self.sampling_formula(u=X[minority_indices[base_ind]],
                                     v=X_maj[nn_maj_ind[base_ind, neigh_ind]],
                                     x=X_min[composite_ind[base_ind, min_neigh_ind]])

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

        # Fitting the nearest neighbors models to the minority and
        # majority data using two different metrics for the minority
        nn_min_euc = NearestNeighbors(n_neighbors=len(X_min),
                                      n_jobs=self.n_jobs)
        nn_min_euc.fit(X_min)
        nn_min_euc_ind = nn_min_euc.kneighbors(X_min, return_distance=False)

        nn_min_cos = NearestNeighbors(n_neighbors=len(X_min),
                                      metric='cosine',
                                      n_jobs=self.n_jobs)
        nn_min_cos.fit(X_min)
        nn_min_cos_ind = nn_min_cos.kneighbors(X_min, return_distance=False)

        n_neighbors = np.min([X_maj.shape[0], self.n_neighbors])
        nn_maj = NearestNeighbors(n_neighbors=n_neighbors,
                                  n_jobs=self.n_jobs)
        nn_maj.fit(X_maj)
        nn_maj_ind = nn_maj.kneighbors(X_min, return_distance=False)

        # prepare the composite neighborhood relationship
        composite_ind = (nn_min_euc_ind.argsort(axis=1) \
                            + nn_min_cos_ind.argsort(axis=1)).argsort(axis=1)

        samples = self.generate_samples(X=X, y=y, X_maj=X_maj, X_min=X_min,
                            n_to_sample=n_to_sample, nn_maj_ind=nn_maj_ind,
                            composite_ind=composite_ind)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
