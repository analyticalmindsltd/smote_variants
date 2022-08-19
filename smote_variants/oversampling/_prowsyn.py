"""
This module implements the ProWSyn method.
"""

import numpy as np

from scipy.linalg import circulant

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['ProWSyn']

class ProWSyn(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @InProceedings{prowsyn,
                        author="Barua, Sukarna
                        and Islam, Md. Monirul
                        and Murase, Kazuyuki",
                        editor="Pei, Jian
                        and Tseng, Vincent S.
                        and Cao, Longbing
                        and Motoda, Hiroshi
                        and Xu, Guandong",
                        title="ProWSyn: Proximity Weighted Synthetic
                                        Oversampling Technique for
                                        Imbalanced Data Set Learning",
                        booktitle="Advances in Knowledge Discovery
                                    and Data Mining",
                        year="2013",
                        publisher="Springer Berlin Heidelberg",
                        address="Berlin, Heidelberg",
                        pages="317--328",
                        isbn="978-3-642-37456-2"
                        }
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 L=5,
                 theta=1.0,
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
            ss_params (dict): simplex sampling parameters
            L (int): number of levels
            theta (float): smoothing factor in weight formula
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
        self.check_greater_or_equal(L, "L", 1)
        self.check_greater_or_equal(theta, "theta", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = ss_params
        self.params = {'L': L,
                        'theta': theta}
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
                                  'L': [3, 5, 7],
                                  'theta': [0.1, 1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def generate_samples_in_clusters(self,
                                        X,
                                        Ps, # pylint: disable=invalid-name
                                        weights,
                                        n_to_sample):
        """
        Generate samples within the clusters.

        Args:
            X (np.array): all training vectors
            Ps (np.array): proximity clusters
            weights (np.array): weights of the clusters
            n_to_sample (int): the number of samples to generate
        """
        clusters_selected = self.random_state.choice(len(Ps),
                                                        n_to_sample,
                                                        p=weights)
        cluster_unique, cluster_count = np.unique(clusters_selected,
                                                    return_counts=True)
        #n_dim_original = self.n_dim
        samples = []
        for idx, cluster in enumerate(cluster_unique):
            cluster_vectors = X[Ps[cluster]]
            #self.n_dim = np.min([self.n_dim, cluster_vectors.shape[0]])
            samples.append(self.sample_simplex(X=cluster_vectors,
                                                indices=circulant(np.arange(len(cluster_vectors))),
                                                n_to_sample = cluster_count[idx]))
            #self.n_dim = n_dim_original

        return np.vstack(samples)

        # do the sampling, from each cluster proportionally to the distribution
        #samples = []
        #while len(samples) < n_to_sample:
        #    cluster_idx = self.random_state.choice(
        #        np.arange(len(weights)), p=weights)
        #    if len(Ps[cluster_idx]) > 1:
        #        random_idx1, random_idx2 = self.random_state.choice(
        #            Ps[cluster_idx], 2, replace=False)
        #        #samples.append(self.sample_between_points(
        #        #    X[random_idx1], X[random_idx2]))
        #        samples.append(X[random_idx1] + self.random_state.random_sample()
        #       *(X[random_idx2] - X[random_idx1]))
        #
        #return np.vstack(samples)

    def determine_clusters_proximity_levels(self, X, y):
        """
        Determine the clusters and proximity levels.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels

        Returns:
            list, np.array: the clusters and the proximity levels
        """
        X_maj = X[y == self.maj_label]
        P = np.where(y == self.min_label)[0] # pylint: disable=invalid-name
        Ps = [] # pylint: disable=invalid-name
        proximity_levels = []

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        # Step 3
        for idx in range(self.params['L']):
            if len(P) == 0:
                break
            # Step 3 a
            n_neighbors = np.min([len(P), self.n_neighbors])

            nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
            nnmt.fit(X[P])
            indices = nnmt.kneighbors(X_maj, return_distance=False)

            # Step 3 b
            P_i = np.unique(indices.flatten()) # pylint: disable=invalid-name

            # Step 3 c - proximity levels are encoded in the Ps list index
            Ps.append(P[P_i])
            proximity_levels.append(idx + 1)

            # Step 3 d
            P = np.delete(P, P_i) # pylint: disable=invalid-name

        # Step 4 and 5
        if len(P) > 0:
            Ps.append(P)
            proximity_levels.append(idx + 1)

        proximity_levels = np.array(proximity_levels)

        return Ps, proximity_levels

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and
                                    target labels
        """
        # Step 1 - a bit generalized
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed.")

        # Step 2
        Ps, proximity_levels = self.determine_clusters_proximity_levels(X, y) # pylint: disable=invalid-name

        # Step 6
        weights = np.exp(-self.params['theta']*(proximity_levels-1))

        # I think this can never happen
        #if not np.any(len(Ps_i) > 1 for Ps_i in Ps):
        #    return self.return_copies(X, y,
        #            "No clusters with at leats 2 samples")

        weights[np.array([idx for idx, Ps_i in enumerate(Ps)
                              if len(Ps_i) == 0], dtype=int)] = 0.0

        # weights is the probability distribution of sampling in the
        # clusters identified
        weights = weights/np.sum(weights)

        samples = self.generate_samples_in_clusters(X, Ps, weights,
                                                    n_to_sample)

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
                'L': self.params['L'],
                'theta': self.params['theta'],
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
