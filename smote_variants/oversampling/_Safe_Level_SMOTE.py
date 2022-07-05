import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['Safe_Level_SMOTE']

class Safe_Level_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @inproceedings{safe_level_smote,
                        author = {
                            Bunkhumpornpat, Chumphol and Sinapiromsaran,
                        Krung and Lursinsap, Chidchanok},
                        title = {Safe-Level-SMOTE: Safe-Level-Synthetic
                                Minority Over-Sampling TEchnique for
                                Handling the Class Imbalanced Problem},
                        booktitle = {Proceedings of the 13th Pacific-Asia
                                    Conference on Advances in Knowledge
                                    Discovery and Data Mining},
                        series = {PAKDD '09},
                        year = {2009},
                        isbn = {978-3-642-01306-5},
                        location = {Bangkok, Thailand},
                        pages = {475--482},
                        numpages = {8},
                        url = {http://dx.doi.org/10.1007/978-3-642-01307-2_43},
                        doi = {10.1007/978-3-642-01307-2_43},
                        acmid = {1533904},
                        publisher = {Springer-Verlag},
                        address = {Berlin, Heidelberg},
                        keywords = {Class Imbalanced Problem, Over-sampling,
                                    SMOTE, Safe Level},
                    }

    Notes:
        * The original method was not prepared for the case when no minority
            sample has minority neighbors.
    """

    categories = [OverSampling.cat_borderline,
                  OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params= {},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
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

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1.0)
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

        # fitting nearest neighbors model
        n_neighbors = min([self.n_neighbors+1, len(X)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)

        minority_labels = (y == self.min_label)
        minority_indices = np.where(minority_labels)[0]

        # do the sampling
        numattrs = len(X[0])
        samples = []
        for _ in range(n_to_sample):
            index = self.random_state.randint(len(minority_indices))
            neighbor_index = self.random_state.choice(indices[index][1:])

            p = X[index]
            n = X[neighbor_index]

            # find safe levels
            sl_p = np.sum(y[indices[index][1:]] == self.min_label)
            sl_n = np.sum(y[indices[neighbor_index][1:]]
                          == self.min_label)

            if sl_n > 0:
                sl_ratio = float(sl_p)/sl_n
            else:
                sl_ratio = np.inf

            if sl_ratio == np.inf and sl_p == 0:
                pass
            else:
                s = np.zeros(numattrs)
                for atti in range(numattrs):
                    # iterate through attributes and do sampling according to
                    # safe level
                    if sl_ratio == np.inf and sl_p > 0:
                        gap = 0.0
                    elif sl_ratio == 1:
                        gap = self.random_state.random_sample()
                    elif sl_ratio > 1:
                        gap = self.random_state.random_sample()*1.0/sl_ratio
                    elif sl_ratio < 1:
                        gap = (1 - sl_ratio) + \
                            self.random_state.random_sample()*sl_ratio
                    dif = n[atti] - p[atti]
                    s[atti] = p[atti] + gap*dif
                samples.append(s)

        if len(samples) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "No samples generated")
            return X.copy(), y.copy()
        else:
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
