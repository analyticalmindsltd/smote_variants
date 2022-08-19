"""
This module implements the G_SMOTE method.
"""

import numpy as np

from sklearn.metrics import pairwise_distances

from ..base import coalesce, coalesce_dict
from ..base._simplexsampling import array_array_index
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['G_SMOTE']

class G_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{g_smote,
                            author={Sandhan, T. and Choi, J. Y.},
                            booktitle={2014 22nd International Conference on
                                        Pattern Recognition},
                            title={Handling Imbalanced Datasets by Partially
                                    Guided Hybrid Sampling for Pattern
                                    Recognition},
                            year={2014},
                            volume={},
                            number={},
                            pages={1449-1453},
                            keywords={Gaussian processes;learning (artificial
                                        intelligence);pattern classification;
                                        regression analysis;sampling methods;
                                        support vector machines;imbalanced
                                        datasets;partially guided hybrid
                                        sampling;pattern recognition;real-world
                                        domains;skewed datasets;dataset
                                        rebalancing;learning algorithm;
                                        extremely low minority class samples;
                                        classification tasks;extracted hidden
                                        patterns;support vector machine;
                                        logistic regression;nearest neighbor;
                                        Gaussian process classifier;Support
                                        vector machines;Proteins;Pattern
                                        recognition;Kernel;Databases;Gaussian
                                        processes;Vectors;Imbalanced dataset;
                                        protein classification;ensemble
                                        classifier;bootstrapping;Sat-image
                                        classification;medical diagnoses},
                            doi={10.1109/ICPR.2014.258},
                            ISSN={1051-4651},
                            month={Aug}}

    Notes:
        * the non-linear approach is inefficient
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_componentwise,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 method='linear',
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
            ss_params (dict): the simplex sampling parameters
            method (str): 'linear'/'non-linear_2.0' - the float can be any
                                number: standard deviation in the
                                Gaussian-kernel
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state, checks={'min_n_dim': 2})
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)

        self.sigma = None
        if method != 'linear' and not method.startswith('non-linear'):
            raise ValueError(f"{self.__class__.__name__}: method parameters "\
                                f"{method} is not supported")
        if method.startswith('non-linear'):
            self.sigma = float(method.split('_')[-1])
            if self.sigma <= 0:
                raise ValueError(f"{self.__class__.__name__}: Non-positive "\
                                f"non-linear parameter {self.sigma} is not "\
                                "supported")

        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.method = method
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
                                  'method': ['linear', 'non-linear_0.1',
                                             'non-linear_1.0',
                                             'non-linear_2.0']}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_H(self, X_min): # pylint: disable=invalid-name
        """
        Determine the pricinpal direction H and the kernel

        Args:
            X_min (np.array): the minority vectors

        Returns:
            np.array, callable: the principal direction and the kernel
        """
        if self.method == 'linear':
            # finding H_l by linear decomposition
            cov = np.cov(X_min, rowvar=False)
            eigw, eigv = np.linalg.eig(cov)
            H = eigv[np.argmax(eigw)] # pylint: disable=invalid-name
        else:
            # building a non-linear kernel matrix and finding H_n by its
            # decomposition

            # I believe there are typos in the paper where the functional
            # form of the Gaussian-kernel is specified, the negative sign
            # and the square of the norm has been added.

            kernel_matrix = pairwise_distances(X_min)**2
            kernel_matrix = kernel_matrix/(2.0*self.sigma**2)
            kernel_matrix = np.exp(kernel_matrix)
            eigw, eigv = np.linalg.eig(kernel_matrix)
            H = eigv[np.argmax(eigw)] # pylint: disable=invalid-name

        return H

    def update_indices_by_angles(self,
                                    X,
                                    indices,
                                    H, # pylint: disable=invalid-name
                                    X_min):
        """
        Rearrange the neighborhood indices according to the angles

        Args:
            X (np.array): the base vectors
            indices (np.array): the neighborhoods
            H (np.array): the principal direction
            X_min (np.array): the minority samples used to create the
                                feature space representation in the non-linear
                                case
        Returns:
            np.array: the ordered neighborhoods
        """
        if self.method == 'linear':
            thetas = self.angles(X, indices, H)
        else:
            thetas = self.angles_non_linear(X, indices, H, X_min)
        thetas_argsort = thetas.argsort()
        indices_ordered = array_array_index(indices[:,1:], thetas_argsort)
        thetas_ordered = array_array_index(thetas, thetas_argsort)
        return indices_ordered, thetas_ordered

    def angles(self,
                X,
                indices,
                H # pylint: disable=invalid-name
                ):
        """
        Calculate the angles for each vector in each neighborhood
        in the linear case.

        Args:
            X (np.array): the base vectors
            indices (np.array): the neighborhoods
            H (np.array): the principal direction

        Returns:
            np.array: the angles for each vector in each neighborhood
        """
        P = X[indices[:, 1:]] - X[indices[:, 0]][:, None] # pylint: disable=invalid-name
        inner_product = np.abs(np.einsum('ijk,k->ij', P, H))
        norms = np.linalg.norm(P, axis=2) * np.linalg.norm(H)
        norms[norms == 0.0] = 1e-5

        return np.arccos(inner_product / norms)

    def angles_non_linear(self,
                            X,
                            indices,
                            H, # pylint: disable=invalid-name
                            X_min):
        """
        Calculate the angles in kernel space for each vector in each
        neighborhood in a vectorized form.

        Args:
            X (np.array): base vectors
            indices (np.array): the indices specifying the neighborhoods
            H (np.array): the principal direction
            X_min (np.array): the minority samples used for creating the
                                feature space representation

        Returns:
            np.array: the angles for each vector in each neighborhood
        """
        diff = X[indices[:, 1:]] - X[indices[:, 0]][:, None]
        P = diff[:, None] - X_min[:, None] # pylint: disable=invalid-name
        gram = -np.linalg.norm(P, axis=3)**2 / (2.0 * self.sigma**2)
        inner_product = np.abs(np.einsum('ikj,k->ij', gram, H))
        norm_H = np.linalg.norm(H) # pylint: disable=invalid-name
        norm_gram = np.linalg.norm(gram, axis=1)

        return np.arccos(inner_product / (norm_H * norm_gram))

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

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_min)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        H = self.determine_H(X_min) # pylint: disable=invalid-name

        # generating samples
        indices_ordered, weights = \
                self.update_indices_by_angles(X_min, ind, H, X_min)

        indices_ordered = np.vstack([np.arange(len(indices_ordered)),
                                    indices_ordered[:,:(self.n_dim-1)].T]).T

        weights = np.vstack([np.repeat(1.0, len(weights)),
                            weights[:,:(self.n_dim-1)].T]).T

        samples = self.sample_simplex(X=X_min,
                                        indices=indices_ordered,
                                        n_to_sample=n_to_sample)
        # the weighted case
        #samples = self.sample_simplex(X=X_min,
        #                                indices=indices_ordered,
        #                                n_to_sample=n_to_sample,
        #                                X_vertices=X_min,
        #                                simplex_weights=weights)

        #while len(samples) < n_to_sample:
        #    idx = self.random_state.randint(len(X_min))
            # calculating difference vectors from all neighbors

            #P = X_min[ind[idx][1:]] - X_min[idx]
            #if self.method == 'linear':
            #    # calculating angles with the principal direction
            #    thetas = np.array([self.angle(P, n, H) for n in range(len(P))])
            #else:
            #    thetas = []
            #    # calculating angles of the difference vectors and the
            #    # principal direction in feature space
            #    for n in range(len(P)):
            #        # calculating representation in feature space
            #        feature_vector = np.array([kernel(X_min[k], P[n]) for k in range(len(X_min))])
            #        dp = np.dot(H, feature_vector)
            #        denom = np.linalg.norm(feature_vector)*np.linalg.norm(H)
            #        thetas.append(np.arccos(np.abs(dp)/denom))
            #    thetas = np.array(thetas)

            # using the neighbor with the difference along the most similar
            # direction to the principal direction of the data
            #n = np.argmin(thetas)
        #    X_a = X_min[idx]
            #X_b = X_min[ind[idx][1:][n]]
        #    X_b = X_min[indices_ordered[idx][0]]
        #    samples.append(self.sample_between_points_componentwise(X_a, X_b))

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
                'method': self.method,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
