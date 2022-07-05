import numpy as np

from sklearn.manifold import LocallyLinearEmbedding

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['LLE_SMOTE']

class LLE_SMOTE(OverSampling):
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

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 n_components=2,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 2)
        self.check_greater_or_equal(n_components, 'n_components', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.nn_params = nn_params
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
                                  'n_neighbors': [3, 5, 7],
                                  'n_components': [2, 3, 5]}
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

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        # extracting minority samples
        X_min = X[y == self.min_label]

        # do the locally linear embedding
        lle = LocallyLinearEmbedding(n_neighbors=self.n_neighbors, 
                                        n_components=self.n_components, 
                                        n_jobs=self.n_jobs)
        try:
            lle.fit(X_min)
        except Exception as e:
            return X.copy(), y.copy()
        X_min_transformed = lle.transform(X_min)

        # fitting the nearest neighbors model for sampling
        n_neighbors = min([self.n_neighbors+1, len(X_min_transformed)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, 
                                                                        lle.transform(X), 
                                                                        y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min_transformed)
        ind = nn.kneighbors(X_min_transformed, return_distance=False)

        def solve_for_weights(xi, Z):
            """
            Solve for locally linear embedding weights

            Args:
                xi (np.array): vector
                Z (np.matrix): matrix of neighbors in rows

            Returns:
                np.array: reconstruction weights

            Following https://cs.nyu.edu/~roweis/lle/algorithm.html
            """
            Z = Z - xi
            Z = Z.T
            C = np.dot(Z.T, Z)
            try:
                w = np.linalg.solve(C, np.repeat(1.0, len(C)))
                if np.linalg.norm(w) > 1e8:
                    w = np.repeat(1.0, len(C))
            except Exception as e:
                w = np.repeat(1.0, len(C))
            return w/np.sum(w)

        # generating samples
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X_min))
            random_coords = self.random_state.choice(ind[idx][1:])
            xi = self.sample_between_points(X_min_transformed[idx],
                                            X_min_transformed[random_coords])
            Z = X_min_transformed[ind[idx][1:]]
            w = solve_for_weights(xi, Z)
            samples.append(np.dot(w, X_min[ind[idx][1:]]))

        return (np.vstack([X, np.vstack(samples)]),
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
                'random_state': self._random_state_init}
