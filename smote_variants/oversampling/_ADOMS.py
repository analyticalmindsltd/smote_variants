import numpy as np

from sklearn.decomposition import PCA

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['ADOMS']

class ADOMS(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{adoms,
                            author={Tang, S. and Chen, S.},
                            booktitle={2008 International Conference on
                                        Information Technology and
                                        Applications in Biomedicine},
                            title={The generation mechanism of synthetic
                                    minority class examples},
                            year={2008},
                            volume={},
                            number={},
                            pages={444-447},
                            keywords={medical image processing;
                                        generation mechanism;synthetic
                                        minority class examples;class
                                        imbalance problem;medical image
                                        analysis;oversampling algorithm;
                                        Principal component analysis;
                                        Biomedical imaging;Medical
                                        diagnostic imaging;Information
                                        technology;Biomedical engineering;
                                        Noise generators;Concrete;Nearest
                                        neighbor searches;Data analysis;
                                        Image analysis},
                            doi={10.1109/ITAB.2008.4570642},
                            ISSN={2168-2194},
                            month={May}}
    """

    categories = [OverSampling.cat_dim_reduction,
                  OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and
                                n_min to sample e.g. 1.0 means that after
                                sampling the number of minority samples
                                will be equal to the number of majority
                                samples
            n_neighbors (int): parameter of the nearest neighbor component
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
        self.check_greater_or_equal(proportion, 'proportion', 0.0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    @classmethod
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

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params))
        nn.fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)

        samples = []
        for _ in range(n_to_sample):
            index = self.random_state.randint(len(X_min))
            neighbors = X_min[indices[index]]

            # fitting the PCA
            pca = PCA(n_components=1)
            pca.fit(neighbors)

            # extracting the principal direction
            principal_direction = pca.components_[0]

            # do the sampling according to the description in the paper
            random_index = self.random_state.randint(1, len(neighbors))
            random_neighbor = neighbors[random_index]
            d = np.linalg.norm(random_neighbor - X_min[index])
            r = self.random_state.random_sample()
            inner_product = np.dot(random_neighbor - X_min[index],
                                   principal_direction)
            sign = 1.0 if inner_product > 0.0 else -1.0
            samples.append(X_min[index] + sign*r*d*principal_direction)

        return (np.vstack([X, np.vstack(samples)]),
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
                'random_state': self._random_state_init}
