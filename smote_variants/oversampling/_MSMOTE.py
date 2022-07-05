import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['MSMOTE']

class MSMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{msmote,
                             author = {Hu, Shengguo and Liang,
                                 Yanfeng and Ma, Lintao and He, Ying},
                             title = {MSMOTE: Improving Classification
                                        Performance When Training Data
                                        is Imbalanced},
                             booktitle = {Proceedings of the 2009 Second
                                            International Workshop on
                                            Computer Science and Engineering
                                            - Volume 02},
                             series = {IWCSE '09},
                             year = {2009},
                             isbn = {978-0-7695-3881-5},
                             pages = {13--17},
                             numpages = {5},
                             url = {https://doi.org/10.1109/WCSE.2009.756},
                             doi = {10.1109/WCSE.2009.756},
                             acmid = {1682710},
                             publisher = {IEEE Computer Society},
                             address = {Washington, DC, USA},
                             keywords = {imbalanced data, over-sampling,
                                        SMOTE, AdaBoost, samples groups,
                                        SMOTEBoost},
                            }

    Notes:
        * The original method was not prepared for the case when all
            minority samples are noise.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_borderline,
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
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor
                                component
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
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
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

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # fitting the nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        indices = nn.kneighbors(X_min, return_distance=False)

        noise_mask = np.repeat(False, len(X_min))

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            index = self.random_state.randint(len(X_min))

            n_p = np.sum(y[indices[index][1:]] == self.min_label)

            if n_p == self.n_neighbors:
                sample_type = 'security'
            elif n_p == 0:
                sample_type = 'noise'
                noise_mask[index] = True
                if np.all(noise_mask):
                    _logger.info("All minority samples are noise")
                    return X.copy(), y.copy()
            else:
                sample_type = 'border'

            if sample_type == 'security':
                neighbor_index = self.random_state.choice(indices[index][1:])
            elif sample_type == 'border':
                neighbor_index = indices[index][1]
            else:
                continue

            s_gen = self.sample_between_points_componentwise(X_min[index],
                                                             X[neighbor_index])
            samples.append(s_gen)

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
