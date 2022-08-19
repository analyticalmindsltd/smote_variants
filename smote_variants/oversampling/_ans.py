"""
This module implements the ANS method.
"""

import numpy as np

from ..base import coalesce_dict, coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['ANS']

class ANS(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @article{ans,
                     author = {Siriseriwan, W and Sinapiromsaran, Krung},
                     year = {2017},
                     month = {09},
                     pages = {565-576},
                     title = {Adaptive neighbor synthetic minority OverSamplingSimplexSimplex
                                technique under 1NN outcast handling},
                     volume = {39},
                     booktitle = {Songklanakarin Journal of Science and
                                    Technology}
                     }

    Notes:
        * The method is not prepared for the case when there is no c satisfying
            the condition in line 25 of the algorithm, fixed.
        * The method is not prepared for empty Pused sets, fixed.
    """
    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary,
                  OverSamplingSimplex.cat_density_based,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
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
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling params
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
        parameter_combinations = {'proportion': [
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_C(self, out_border, X): # pylint: disable=invalid-name
        """
        Determines the best C value.

        Args:
            out_border (np.array): outer border
            X (np.array): feature vectors

        Returns:
            float: the best C value
        """
        C_max = int(0.25*len(X)) # pylint: disable=invalid-name
        n_oc_m1 = -1
        C = 0 # pylint: disable=invalid-name
        best_diff = np.inf
        for c in range(1, C_max): # pylint: disable=invalid-name
            n_oc = np.sum(out_border >= c)
            if abs(n_oc - n_oc_m1) < best_diff:
                best_diff = abs(n_oc - n_oc_m1)
                C = n_oc # pylint: disable=invalid-name
            n_oc_m1 = n_oc

        return C

    def determine_first_pos_neighbor_distances(self, X_min, nn_params):
        """
        Determine the first positive nearest neighbor distances.

        Args:
            np.array (X_min): minority (positive) samples
            nn_params (dict): nearest neighbor parameters

        Returns:
            np.array: the first positive neighbor distances
        """
        # finding the first minority neighbor of minority samples
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=2,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nearestn.fit(X_min)
        dist, _ = nearestn.kneighbors(X_min)

        # extracting the distances of first minority neighbors from minority
        # samples
        first_pos_neighbor_distances = dist[:, 1]

        return first_pos_neighbor_distances

    def determine_outer_border(self, *, X, y, X_min,
                                first_pos_neighbor_distances,
                                nn_params):
        """
        Determine the outer border

        Args:
            X (np.array): features
            y (np.array): targets
            X_min (np.array): minority samples
            first_pos_neighbor_distances (np.array): closest positive neighbor
                                                        distances
            nn_params (dict): nearest neighbors parameters

        Returns:
            np.array: the outer border
        """
        # fitting another nearest neighbors model to extract majority
        # samples in the neighborhoods of minority samples
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=1,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nearestn.fit(X)

        # extracting the number of majority samples in the neighborhood of
        # minority samples
        out_border = []
        for idx, row in enumerate(X_min):
            ind = nearestn.radius_neighbors(row.reshape(1, -1),
                                      first_pos_neighbor_distances[idx],
                                      return_distance=False)
            out_border.append(np.sum(y[ind[0].astype(int)] == self.maj_label))

        out_border = np.array(out_border)

        return out_border

    def determine_Pused(self, X, y, X_min, nn_params): # pylint: disable=invalid-name
        """
        Determine the Pused array.

        Args:
            X (np.array): all features
            y (np.array): all targets
            X_min (np.array): minority samples
            nn_params (dict): nearest neighbor parameters

        Returns:
            np.array, float: the Pused array, and the eps value
        """
        first_pos_dist = \
            self.determine_first_pos_neighbor_distances(X_min, nn_params)

        out_border = self.determine_outer_border(X=X, y=y, X_min=X_min,
                                first_pos_neighbor_distances=first_pos_dist,
                                nn_params=nn_params)

        # finding the optimal C value by comparing the number of outcast
        # minority samples when traversing the range [1, C_max]
        # maximum C value
        C = self.determine_C(out_border, X) # pylint: disable=invalid-name

        # determining the set of minority samples Pused
        Pused = np.where(out_border < C)[0] # pylint: disable=invalid-name

        # finding the maximum distances of first positive neighbors
        if Pused.shape[0] == 0:
            eps = None
        else:
            eps = np.max(first_pos_dist[Pused])

        return Pused, eps

    def determine_simplex_weights(self,
                                    ind,
                                    nearestn,
                                    X_min,
                                    Pused # pylint: disable=invalid-name
                                    ):
        """
        Determine simplex weights.

        Args:
            ind (np.array): neighborhood structure
            nearestn (NearestNeighbors): a fitted nearest neighbors object
            X_min (np.array): minority samples
            Pused (np.array): the Pused array

        Returns:
            weights (np.array): the simplex node weights for sampling
        """
        max_neighbors = np.max([len(row) for i, row in enumerate(ind)])
        ind_dense = nearestn.kneighbors(X_min[Pused], max_neighbors, return_distance=False)

        weights = ind_dense.copy()
        weights[:,:]= 1.0

        for idx, row in enumerate(ind):
            weights[idx, len(row):] = 0.0

        return ind_dense, weights

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
            return self.return_copies(X, y, "Sampling is not needed.")

        X_min = X[y == self.min_label]

        # outcast extraction algorithm

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(
                                                            nn_params, X, y)

        Pused, eps = self.determine_Pused(X, y, X_min, nn_params) # pylint: disable=invalid-name

        # Adaptive neighbor SMOTE algorithm

        # checking if there are minority samples left
        if len(Pused) == 0:
            return self.return_copies(X, y, "Pused is empty")

        # fitting nearest neighbors model to find nearest minority samples in
        # the neighborhoods of minority samples
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=1,
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nearestn.fit(X_min[Pused])
        ind = nearestn.radius_neighbors(X_min[Pused],
                                        eps,
                                        return_distance=False)

        # extracting the number of positive samples in the neighborhoods
        Np = np.array([len(i) for i in ind]) # pylint: disable=invalid-name

        if np.all(Np == 1):
            return self.return_copies(X, y, "all samples have only 1 neighbor"\
                                                        " in the given radius")

        # determining the distribution used to generate samples
        Np = Np/np.sum(Np) # pylint: disable=invalid-name

        ind_dense, weights = self.determine_simplex_weights(ind, nearestn,
                                                            X_min, Pused)

        samples = self.sample_simplex(X=X_min[Pused],
                                        indices=ind_dense,
                                        n_to_sample=n_to_sample,
                                        simplex_weights=weights,
                                        base_weights=Np)

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, samples.shape[0])]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
