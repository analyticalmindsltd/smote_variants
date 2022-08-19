"""
This module implements the V_SYNTH method.
"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import scipy.spatial as sspatial

from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['V_SYNTH']

class V_SYNTH(OverSampling):
    """
    References:
        * BibTex::

            @article{v_synth,
                     author = {Young,Ii, William A. and Nykl, Scott L. and
                                Weckman, Gary R. and Chelberg, David M.},
                     title = {Using Voronoi Diagrams to Improve
                                Classification Performances when Modeling
                                Imbalanced Datasets},
                     journal = {Neural Comput. Appl.},
                     issue_date = {July      2015},
                     volume = {26},
                     number = {5},
                     month = jul,
                     year = {2015},
                     issn = {0941-0643},
                     pages = {1041--1054},
                     numpages = {14},
                     url = {http://dx.doi.org/10.1007/s00521-014-1780-0},
                     doi = {10.1007/s00521-014-1780-0},
                     acmid = {2790665},
                     publisher = {Springer-Verlag},
                     address = {London, UK, UK},
                     keywords = {Data engineering, Data mining, Imbalanced
                                    datasets, Knowledge extraction,
                                    Numerical algorithms, Synthetic
                                    over-sampling},
                    }

    Notes:
        * The proposed encompassing bounding box generation is incorrect.
        * Voronoi diagram generation in high dimensional spaces is instable
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 n_components=3,
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
            n_components (int): number of components for PCA
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state, checks={'min_n_dim': 2})
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_components, "n_component", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
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
                                  'n_components': [3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def add_bounding_box(self, X):
        """
        Add bounding box

        Args:
            X (np.array): all training vectors

        Returns:
            np.array, np.array: the extended set of vectors and the
                                bounding box
        """
        # creating the bounding box
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        mins = mins - 0.1*np.abs(mins)
        maxs = maxs + 0.1*np.abs(maxs)

        dim = len(X[0])

        def random_min_maxs():
            return np.where(self.random_state.randint(0, 1, size=dim) == 0,
                            mins,
                            maxs)

        n_bounding_box = min([100, len(X[0])])
        bounding_box = [random_min_maxs() for _ in range(n_bounding_box)]
        X_bb = np.vstack([X, bounding_box]) # pylint: disable=invalid-name

        return X_bb, bounding_box

    def filter_vectors_too_close(self,
                                X_pca, # pylint: disable=invalid-name
                                y_pca):
        """
        Determine the Voronoi-tessellation

        Args:
            X_pca (np.array): the vectors after PCA
            y_pca (np.array): the extended target labels

        Returns:
            np.array, np.array: the updated training vectors
                                and target labels
        """
        distm = pairwise_distances(X_pca)

        to_remove = []
        for idx in range(len(distm)):
            for jdx in range(idx+1, len(distm)):
                if distm[idx, jdx] < 0.001:
                    to_remove.append(idx)

        X_pca = np.delete(X_pca, to_remove, axis=0)
        y_pca = np.delete(y_pca, to_remove)

        return X_pca, y_pca

    def generate_samples(self, *,
                            voronoi,
                            y_pca,
                            candidate_face_generators,
                            n_to_sample):
        """
        Generate samples.

        Args:
            voronoi (obj): a Voronoi tessellation
            y_pca (np.array): the updated target labels
            candidate_face_generators (np.array): indices of the
                                                    face generators
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        samples = []
        for _ in range(n_to_sample):
            # randomly choosing a pair from the ridge point pairs of different
            # labels
            random_face = self.random_state.choice(candidate_face_generators)

            # extracting the vertices of the face between the points
            ridge_vertices = voronoi.ridge_vertices[random_face]
            face_vertices = voronoi.vertices[ridge_vertices]

            # creating a random vector for sampling the face (supposed to be
            # convex)
            weights = self.random_state.random_sample(size=len(face_vertices))
            weights = weights / np.sum(weights)

            # initiating a sample point on the face
            sample_point_on_face = np.sum((face_vertices.T * weights).T, axis=0)

            # finding the ridge point with the minority label
            if y_pca[voronoi.ridge_points[random_face][0]] == self.min_label:
                h_vec = voronoi.points[voronoi.ridge_points[random_face][0]]
            else:
                h_vec = voronoi.points[voronoi.ridge_points[random_face][1]]

            # generating a point between the minority ridge point and the
            # random point on the face
            samples.append(self.sample_between_points(sample_point_on_face,
                                                      h_vec))
        return np.vstack(samples)

    def dimensionality_reduction(self,
                                    X_bb, # pylint: disable=invalid-name
                                    y,
                                    bounding_box):
        """
        Do the dimensionality reduction.

        Args:
            X_bb (np.array): the extended set of vectors
            y (np.array): the target labels
            bounding_box (np.array): the bounding box vectors

        Returns:
            np.array, np.array, int, obj: the PCA-d training vectors, the
                                    corresponding target labels, the number
                                    of components and the fitted PCA object
        """
        # applying PCA to reduce the dimensionality of the data
        n_components = np.min([X_bb.shape[1], self.n_components])
        component_mask = np.array([True] * n_components)

        while np.sum(component_mask) > 1:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_bb) # pylint: disable=invalid-name
            y_pca = np.hstack([y, np.repeat(-1, len(bounding_box))])

            component_mask = pca.explained_variance_ratio_ > 1e-2

            if np.sum(component_mask) == n_components:
                break

            n_components = np.sum(component_mask)

        return X_pca, y_pca, n_components, pca

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

        X_bb, bounding_box = self.add_bounding_box(X) # pylint: disable=invalid-name

        X_pca, y_pca, n_components, pca = \
                        self.dimensionality_reduction(X_bb, y, bounding_box) # pylint: disable=invalid-name

        if n_components == 1:
            return self.return_copies(X, y,
                                "Voronoi tessellation in 1D is not feasible")

        X_pca, y_pca = self.filter_vectors_too_close(X_pca, y_pca) # pylint: disable=invalid-name

        # doing the Voronoi tessellation
        voronoi = sspatial.Voronoi(X_pca)

        # extracting those ridge point pairs which are candidates for
        # generating an edge between two cells of different class labels
        candidate_face_generators = []
        for idx, ridge in enumerate(voronoi.ridge_points):
            if ridge[0] < len(y) and ridge[1] < len(y) \
                            and y[ridge[0]] != y[ridge[1]]:
                candidate_face_generators.append(idx)

        # seems like this never happens
        if len(candidate_face_generators) == 0:
            return self.return_copies(X, y, "No candidate face generators found")

        # generating samples
        samples = self.generate_samples(voronoi=voronoi,
                                        y_pca=y_pca,
                                        candidate_face_generators=candidate_face_generators,
                                        n_to_sample=n_to_sample)

        return (np.vstack([X, pca.inverse_transform(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_components': self.n_components,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
