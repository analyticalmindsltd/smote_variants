import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['Edge_Det_SMOTE']

class Edge_Det_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{Edge_Det_SMOTE,
                            author={Kang, Y. and Won, S.},
                            booktitle={ICCAS 2010},
                            title={Weight decision algorithm for oversampling
                                    technique on class-imbalanced learning},
                            year={2010},
                            volume={},
                            number={},
                            pages={182-186},
                            keywords={edge detection;learning (artificial
                                        intelligence);weight decision
                                        algorithm;oversampling technique;
                                        class-imbalanced learning;class
                                        imbalanced data problem;edge
                                        detection algorithm;spatial space
                                        representation;Classification
                                        algorithms;Image edge detection;
                                        Training;Noise measurement;Glass;
                                        Training data;Machine learning;
                                        Imbalanced learning;Classification;
                                        Weight decision;Oversampling;
                                        Edge detection},
                            doi={10.1109/ICCAS.2010.5669889},
                            ISSN={},
                            month={Oct}}

    Notes:
        * This technique is very loosely specified.
    """

    categories = [OverSampling.cat_density_based,
                  OverSampling.cat_borderline,
                  OverSampling.cat_extensive,
                  OverSampling.cat_metric_learning]

    def __init__(self, 
                 proportion=1.0, 
                 k=5, 
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
            k (int): number of neighbors
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
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, "k", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k
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
                                  'k': [3, 5, 7]}
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        d = len(X[0])
        X_min = X[y == self.min_label]

        # organizing class labels according to feature ranking
        magnitudes = np.zeros(len(X))
        for i in range(d):
            to_sort = zip(X[:, i], np.arange(len(X)), y)
            _, idx, label = zip(*sorted(to_sort, key=lambda x: x[0]))
            # extracting edge magnitudes in this dimension
            for j in range(1, len(idx)-1):
                magnitudes[idx[j]] = magnitudes[idx[j]] + \
                    (label[j-1] - label[j+1])**2

        # density estimation
        magnitudes = magnitudes[y == self.min_label]
        magnitudes[magnitudes < 0]= 0
        magnitudes= np.nan_to_num(magnitudes, nan=0.0)
        if np.sum(magnitudes) == 0:
            magnitudes= np.repeat(1.0/len(magnitudes), len(magnitudes))
        magnitudes = np.sqrt(magnitudes)
        magnitudes = magnitudes/np.sum(magnitudes)

        # fitting nearest neighbors models to minority samples
        n_neighbors = min([len(X_min), self.k+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        ind = nn.kneighbors(X_min, return_distance=False)

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.choice(np.arange(len(X_min)), p=magnitudes)
            X_a = X_min[idx]
            X_b = X_min[self.random_state.choice(ind[idx][1:])]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

