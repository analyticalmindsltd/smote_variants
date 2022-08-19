"""
This module implements the LLE_SMOTE method.
"""

import numpy as np
from numpy.linalg import LinAlgError

from sklearn.manifold import LocallyLinearEmbedding

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['LLE_SMOTE']

class LLE_SMOTE(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{lle_smote,
                            author={Wang, J. and Xu, M. and Wang,
                                    H. and Zhang, J.},
                            booktitle={2006 8th international Conference
                                    on Signal Processing},
                            title={Classification of Imbalanced Data by Using
                                    the SMOTE Algorithm and Locally Linear
                                    Embedding},
                            year={2006},
                            volume={3},
                            number={},
                            pages={},
                            keywords={artificial intelligence;
                                        biomedical imaging;medical computing;
                                        imbalanced data classification;
                                        SMOTE algorithm;
                                        locally linear embedding;
                                        medical imaging intelligence;
                                        synthetic minority oversampling
                                        technique;
                                        high-dimensional data;
                                        low-dimensional space;
                                        Biomedical imaging;
                                        Back;Training data;
                                        Data mining;Biomedical engineering;
                                        Research and development;
                                        Electronic mail;Pattern recognition;
                                        Performance analysis;
                                        Classification algorithms},
                            doi={10.1109/ICOSP.2006.345752},
                            ISSN={2164-5221},
                            month={Nov}}

    Notes:
        * There might be numerical issues if the nearest neighbors contain
            some element multiple times.
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_dim_reduction,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 n_components=2,
                 nn_params=None,
                 ss_params=None,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj
                                and n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples will
                                be equal to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
            n_components (int): dimensionality of the embedding space
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        nn_params = coalesce(nn_params, {})

        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 2)
        self.check_greater_or_equal(n_components, 'n_components', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_components = n_components
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
                                  'n_neighbors': [3, 5, 7],
                                  'n_components': [2, 3, 5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def solve_for_weights(self,
                            xi, # pylint: disable=invalid-name
                            Z): # pylint: disable=invalid-name
        """
        Solve for locally linear embedding weights

        Args:
            xi (np.array): vector
            Z (np.array): matrix of neighbors in rows

        Returns:
            np.array: reconstruction weights

        Following https://cs.nyu.edu/~roweis/lle/algorithm.html
        """
        Z = Z - xi # pylint: disable=invalid-name
        Z = Z.T # pylint: disable=invalid-name
        C = np.dot(Z.T, Z) # pylint: disable=invalid-name
        try:
            weights = np.linalg.solve(C, np.repeat(1.0, C.shape[0]))
            if np.linalg.norm(weights) > 1e8:
                weights = np.repeat(1.0, C.shape[0])
        except LinAlgError:
            weights = np.repeat(1.0, C.shape[0])

        return weights/np.sum(weights)

    def embed_in_original_space(self,
                                X_min_transformed, # pylint: disable=invalid-name
                                X_min,
                                nnmt,
                                samples_raw):
        """
        Carry out the embedding in the original feature space

        Args:
            X_min_transformed (np.array): the minority samples after LLE
            X_min (np.array): the original minority samples
            nnmt (obj): fitted nearest neighbors object
            samples_raw (np.array): the raw samples

        Returns:
            np.array: the final samples
        """
        ind = nnmt.kneighbors(samples_raw, return_distance=False)

        samples = []
        for idx, sample in enumerate(samples_raw):
            weights = self.solve_for_weights(sample, X_min_transformed[ind[idx]])
            samples.append(np.dot(weights, X_min[ind[idx]]))

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

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # extracting minority samples
        X_min = X[y == self.min_label]

        n_components = np.min([self.n_components, X.shape[1]])
        n_neighbors = np.min([self.n_neighbors, X_min.shape[0] - 1])

        # do the locally linear embedding
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                        n_components=n_components,
                                        n_jobs=self.n_jobs)
        lle.fit(X_min)
        X_min_transformed = lle.transform(X_min) # pylint: disable=invalid-name

        # fitting the nearest neighbors model for sampling
        n_neighbors = min([self.n_neighbors + 1, len(X_min_transformed)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params,
                                                                        lle.transform(X),
                                                                        y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_min_transformed)
        ind = nnmt.kneighbors(X_min_transformed, return_distance=False)

        samples_raw = self.sample_simplex(X=X_min_transformed,
                                        indices=ind,
                                        n_to_sample=n_to_sample)

        samples = self.embed_in_original_space(X_min_transformed, # pylint: disable=invalid-name
                                                X_min,
                                                nnmt,
                                                samples_raw)

        # generating samples
        #samples = []
        #for _ in range(n_to_sample):
        #    idx = self.random_state.randint(len(X_min))
        #    random_coords = self.random_state.choice(ind[idx][1:])
        #    xi = self.sample_between_points(X_min_transformed[idx],
        #                                    X_min_transformed[random_coords])
        #    Z = X_min_transformed[ind[idx][1:]]
        #    w = solve_for_weights(xi, Z)
        #    samples.append(np.dot(w, X_min[ind[idx][1:]]))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'n_components': self.n_components,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
