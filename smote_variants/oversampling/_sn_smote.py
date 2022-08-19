"""
This module implements the SNSMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import (NearestNeighborsWithMetricTensor,
                            distances_mahalanobis)
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['SN_SMOTE']

class SN_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @Article{sn_smote,
                        author="Garc{\'i}a, V.
                        and S{\'a}nchez, J. S.
                        and Mart{\'i}n-F{\'e}lez, R.
                        and Mollineda, R. A.",
                        title="Surrounding neighborhood-based SMOTE for
                                learning from imbalanced data sets",
                        journal="Progress in Artificial Intelligence",
                        year="2012",
                        month="Dec",
                        day="01",
                        volume="1",
                        number="4",
                        pages="347--362",
                        issn="2192-6360",
                        doi="10.1007/s13748-012-0027-5",
                        url="https://doi.org/10.1007/s13748-012-0027-5"
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
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (float): number of neighbors
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
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
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

    def generate_centroid_neighborhood(self, X_min, nn_params, ind):
        """
        Generate the centroid neighborhood structure.

        Args:
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbor parameters
            ind (np.array): the neighborhood structure

        """
        n_neighbors = ind.shape[1]

        # determining k nearest centroid neighbors
        ncn = np.zeros(shape=(X_min.shape[0], self.n_neighbors)).astype(int)

        metric_tensor = coalesce(nn_params.get('metric_tensor', None),
                                 np.eye(X_min.shape[1]))

        ncn[:, 0] = ind[:, 1]
        n_cents = np.ones(ncn.shape[0]).astype(int)
        centroids = X_min[ncn[:, 0]]
        cent_dist = distances_mahalanobis(X=X_min,
                                            Y=centroids,
                                            tensor=metric_tensor)

        mask = np.repeat(True, X_min.shape[0])
        jdx = 2
        while np.any(mask):
            centroids_tmp = ((centroids[mask] + X_min[ind[mask, jdx]]).T / (n_cents[mask] + 1)).T
            new_cent_dist = distances_mahalanobis(X=X_min[mask],
                                                    Y=centroids_tmp,
                                                    tensor=metric_tensor)

            improved = new_cent_dist < cent_dist[mask]
            centroids[mask][improved] = centroids[mask][improved] + X_min[ind[mask, jdx][improved]]
            ncn[mask][improved, n_cents[mask][improved]] = ind[mask][improved, jdx]
            n_cents[mask][improved] = n_cents[mask][improved] + 1
            cent_dist[mask][improved] = new_cent_dist[improved]

            mask = (n_cents < self.n_neighbors)

            jdx = jdx + 1
            if jdx == n_neighbors:
                break

        return ncn, n_cents

    def generate_samples(self, X_min, n_to_sample, ncn, ncn_nums):
        """
        Generate the samples.

        Args:
            X_min (np.array): minortity samples
            n_to_sample (int): the number of samples to generate
            ncn (np.array): the nearest centroid neighborhood
            ncn_nums (np.array): the sizes of the neighborhoods

        Returns:
            np.array: the generated samples
        """
        samples = np.zeros(shape=(0, X_min.shape[1]))

        base_indices = self.random_state.choice(X_min.shape[0], n_to_sample)
        base_unique, base_count = np.unique(base_indices, return_counts=True)

        for idx, base_idx in enumerate(base_unique):
            X_vertices = X_min[ncn[base_idx, :ncn_nums[base_idx]]]
            indices = np.array([np.hstack([np.array([0]),
                                np.arange(ncn_nums[base_idx])])])

            #n_dim_orig = self.n_dim
            #self.n_dim = np.min([ncn_nums[base_idx] + 1, n_dim_orig])

            samples_tmp = self.sample_simplex(X=X_min[[base_idx]],
                                                indices=indices,
                                                n_to_sample=base_count[idx],
                                                X_vertices=X_vertices)
            #self.n_dim = n_dim_orig

            samples = np.vstack([samples, samples_tmp])

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
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        X_min = X[y == self.min_label]

        # the search for the k nearest centroid neighbors is limited for the
        # nearest 10*n_neighbors neighbors

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        n_neighbors = min([self.n_neighbors*10, len(X_min)])
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        ncn, ncn_nums = self.generate_centroid_neighborhood(X_min, nn_params, ind)

        # extracting nearest centroid neighbors
        #for idx, x_min in enumerate(X_min):
        #    # the first NCN neighbor is the first neighbor
        #    ncn[idx, 0] = ind[idx][1]
        #
        #    # iterating through all neighbors and finding the one with smaller
        #    # centroid distance to X_min[i] than the previous set of neighbors
        #    n_cent = 1
        #    centroid = X_min[ncn[idx, 0]]
        #
        #    cent_dist = np.sqrt(np.dot(np.dot((centroid - x_min), metric_tensor),
        #                                (centroid - x_min)))
        #    jdx = 2
        #    while jdx < len(ind[idx]) and n_cent < self.n_neighbors:
        #        diff_vect = (centroid + X_min[ind[idx][jdx]])/(n_cent + 1) - x_min
        #        new_cent_dist = np.sqrt(np.dot(np.dot(diff_vect, metric_tensor), diff_vect))
        #
        #        # checking if new nearest centroid neighbor found
        #        if new_cent_dist < cent_dist:
        #            centroid = centroid + X_min[ind[idx][jdx]]
        #            ncn[idx, n_cent] = ind[idx][jdx]
        #            n_cent = n_cent + 1
        #            cent_dist = new_cent_dist
        #
        #        jdx = jdx + 1
        #
        #    # registering the number of nearest centroid neighbors found
        #    ncn_nums[idx] = n_cent

        # generating samples
        #samples = []
        #while len(samples) < n_to_sample:
        #    random_idx = self.random_state.randint(len(X_min))
        #    random_neighbor_idx = self.random_state.choice(
        #        ncn[random_idx][:ncn_nums[random_idx]])
        #    samples.append(self.sample_between_points(
        #        X_min[random_idx], X_min[random_neighbor_idx]))

        samples = self.generate_samples(X_min, n_to_sample, ncn, ncn_nums)

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
                **OverSamplingSimplex.get_params(self)}
