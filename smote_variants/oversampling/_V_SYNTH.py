import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import scipy.spatial as sspatial

from ._OverSampling import OverSampling
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
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_components, "n_component", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_components = n_components
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

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
        bounding_box = [random_min_maxs() for i in range(n_bounding_box)]
        X_bb = np.vstack([X, bounding_box])

        # applying PCA to reduce the dimensionality of the data
        n_components = min([len(X[0]), self.n_components])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_bb)
        y_pca = np.hstack([y, np.repeat(-1, len(bounding_box))])

        dm = pairwise_distances(X_pca)
        to_remove = []
        for i in range(len(dm)):
            for j in range(i+1, len(dm)):
                if dm[i, j] < 0.001:
                    to_remove.append(i)
        X_pca = np.delete(X_pca, to_remove, axis=0)
        y_pca = np.delete(y_pca, to_remove)

        # doing the Voronoi tessellation
        voronoi = sspatial.Voronoi(X_pca)

        # extracting those ridge point pairs which are candidates for
        # generating an edge between two cells of different class labels
        candidate_face_generators = []
        for i, r in enumerate(voronoi.ridge_points):
            if r[0] < len(y) and r[1] < len(y) and not y[r[0]] == y[r[1]]:
                candidate_face_generators.append(i)

        if len(candidate_face_generators) == 0:
            return X.copy(), y.copy()

        # generating samples
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
            w = self.random_state.random_sample(size=len(X_pca[0]))
            w = w/np.sum(w)

            # initiating a sample point on the face
            sample_point_on_face = np.zeros(len(X_pca[0]))
            for i in range(len(X_pca[0])):
                sample_point_on_face += w[i]*face_vertices[i]

            # finding the ridge point with the minority label
            if y[voronoi.ridge_points[random_face][0]] == self.min_label:
                h = voronoi.points[voronoi.ridge_points[random_face][0]]
            else:
                h = voronoi.points[voronoi.ridge_points[random_face][1]]

            # generating a point between the minority ridge point and the
            # random point on the face
            samples.append(self.sample_between_points(sample_point_on_face,
                                                      h))

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
                'random_state': self._random_state_init}

