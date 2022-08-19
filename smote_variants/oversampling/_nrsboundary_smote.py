"""
This module implements the NRSBoundary_SMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import (NearestNeighborsWithMetricTensor,
                                pairwise_distances_mahalanobis)
from ..base import OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['NRSBoundary_SMOTE']

class NRSBoundary_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @Article{nrsboundary_smote,
                    author= {Feng, Hu and Hang, Li},
                    title= {A Novel Boundary Oversampling Algorithm Based on
                            Neighborhood Rough Set Model: NRSBoundary-SMOTE},
                    journal= {Mathematical Problems in Engineering},
                    year= {2013},
                    pages= {10},
                    doi= {10.1155/2013/694809},
                    url= {http://dx.doi.org/10.1155/694809}
                    }
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_borderline,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 w=0.005, # pylint: disable=invalid-name
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
            w (float): used to set neighborhood radius
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
        self.check_greater_or_equal(w, "w", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.w = w # pylint: disable=invalid-name
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
                                  'w': [0.005, 0.01, 0.05]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_delta(self, X, nn_params):
        """
        Determine the delta values

        Args:
            X (np.array): all training vectors
            nn_params (dict): nearest neighbor parameters

        Returns:
            np.array: the delta values
        """
        distm = pairwise_distances_mahalanobis(X,
                                               tensor=nn_params['metric_tensor'])
        d_max = np.max(distm, axis=1)
        np.fill_diagonal(distm, np.max(distm))
        d_min = np.min(distm, axis=1)

        delta = d_min + self.w*(d_max - d_min)

        return delta

    def determine_bound_pos_set_delta(self, X, y, nn_params):
        """
        Determine bound_set and pos_set and the delta vector.

        Args:
            X (np.array): training vectors
            y (np.array): target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array, np.array: the bound_set, pos_set and delta
        """
        delta = self.determine_delta(X, nn_params)

        # step 1
        bound_set = []
        pos_set = []

        # number of neighbors is not interesting here, as we use the
        # radius_neighbors function to extract the neighbors in a given radius
        n_neighbors = np.min([self.n_neighbors + 1, len(X)])

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        for idx, x_vec in enumerate(X):
            indices = nnmt.radius_neighbors(x_vec.reshape(1, -1),
                                          delta[idx],
                                          return_distance=False)

            n_minority = np.sum(y[indices[0].astype(int)] == self.min_label)
            n_majority = np.sum(y[indices[0].astype(int)] == self.maj_label)
            if y[idx] == self.min_label and not n_minority == len(indices[0]):
                bound_set.append(idx)
            elif y[idx] == self.maj_label and n_majority == len(indices[0]):
                pos_set.append(idx)

        bound_set = np.array(bound_set)
        pos_set = np.array(pos_set)

        return bound_set, pos_set, delta

    def generate_samples(self, *, X, X_min,
                    indices, bound_set, pos_set,
                    delta, n_to_sample):
        """
        Generate samples.

        Args:
            X (np.array): all training vectors
            X_min (np.array): minority vectors
            indices (np.array): the neighborhood structure
            bound_set (np.array): indices of the boundary set
            pos_set (np.array): indices of the positive set
            delta (np.array): delta distances
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the new samples
        """
        trials = 0
        samples = np.zeros(shape=(0, X.shape[1]))

        while len(samples) < n_to_sample and trials < n_to_sample:
            subsample = self.sample_simplex(X=X[bound_set],
                                            indices=indices,
                                            n_to_sample=(n_to_sample - len(samples))*4,
                                            X_vertices=X_min)
            # checking the conflict
            dist_from_pos_set = \
                    np.linalg.norm(subsample - X[pos_set][:, None], axis=2)
            no_conflict = np.all(dist_from_pos_set.T > delta[pos_set], axis=1)
            samples = np.vstack([samples, subsample[no_conflict]])

            trials = trials + 1

        return np.vstack(samples)[:n_to_sample]

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # determining the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed.")

        # step 2
        X_min_indices = np.where(y == self.min_label)[0] # pylint: disable=invalid-name
        X_min = X[X_min_indices]

        # step 3
        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        bound_set, pos_set, delta = \
                    self.determine_bound_pos_set_delta(X, y, nn_params)

        if len(pos_set) == 0 or len(bound_set) == 0:
            return self.return_copies(X, y, "bound set or pos set empty")

        # step 4 and 5
        # computing the nearest neighbors of the bound set from the
        # minority set
        n_neighbors = min([len(X_min), self.n_neighbors + 1])
        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min)
        indices = nnmt.kneighbors(X[bound_set], return_distance=False)

        samples = self.generate_samples(X=X, X_min=X_min,
                        indices=indices, bound_set=bound_set, pos_set=pos_set,
                        delta=delta, n_to_sample=n_to_sample)


        # do the sampling
        #samples = []
        #trials = 0
        #w = self.w
        #while len(samples) < n_to_sample:
        #    idx = self.random_state.choice(len(bound_set))
        #    random_neighbor_idx = self.random_state.choice(indices[idx][1:])
        #    x_new = self.sample_between_points(
        #        X[bound_set[idx]], X_min[random_neighbor_idx])
        #
        #    # checking the conflict
        #    dist_from_pos_set = np.linalg.norm(X[pos_set] - x_new, axis=1)
        #    if np.all(dist_from_pos_set > delta[pos_set]):
        #        # no conflict
        #        samples.append(x_new)
        #    trials = trials + 1
        #    if trials > 1000 and len(samples) == 0:
        #        trials = 0
        #        w = w*0.9

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
                'w': self.w,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
