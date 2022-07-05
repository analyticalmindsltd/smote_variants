import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['LN_SMOTE']

class LN_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{ln_smote,
                            author={Maciejewski, T. and Stefanowski, J.},
                            booktitle={2011 IEEE Symposium on Computational
                                        Intelligence and Data Mining (CIDM)},
                            title={Local neighbourhood extension of SMOTE for
                                        mining imbalanced data},
                            year={2011},
                            volume={},
                            number={},
                            pages={104-111},
                            keywords={Bayes methods;data mining;pattern
                                        classification;local neighbourhood
                                        extension;imbalanced data mining;
                                        focused resampling technique;SMOTE
                                        over-sampling method;naive Bayes
                                        classifiers;Noise measurement;Noise;
                                        Decision trees;Breast cancer;
                                        Sensitivity;Data mining;Training},
                            doi={10.1109/CIDM.2011.5949434},
                            ISSN={},
                            month={April}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
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
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
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
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
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

        # number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        if self.n_neighbors + 2 > len(X):
            n_neighbors = len(X) - 2
        else:
            n_neighbors = self.n_neighbors

        if n_neighbors < 2:
            return X.copy(), y.copy()

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # nearest neighbors of each instance to each instance in the dataset
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors + 2,
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        indices = nn.kneighbors(X, return_distance=False)

        minority_indices = np.where(y == self.min_label)[0]

        # dimensionality
        d = len(X[0])

        def safe_level(p_idx, n_idx=None):
            """
            computing the safe level of samples

            Args:
                p_idx (int): index of positive sample
                n_idx (int): index of other sample

            Returns:
                int: safe level
            """
            if n_idx is None:
                # implementation for 1 sample only
                return np.sum(y[indices[p_idx][1:-1]] == self.min_label)
            else:
                # implementation for 2 samples
                if ((not y[n_idx] != self.maj_label)
                        and p_idx in indices[n_idx][1:-1]):
                    # -1 because p_idx will be replaced
                    n_positives = np.sum(
                        y[indices[n_idx][1:-1]] == self.min_label) - 1
                    if y[indices[n_idx][-1]] == self.min_label:
                        # this is the effect of replacing p_idx by the next
                        # (k+1)th neighbor
                        n_positives = n_positives + 1
                    return n_positives
                return np.sum(y[indices[n_idx][1:-1]] == self.min_label)

        def random_gap(slp, sln, n_label):
            """
            determining random gap

            Args:
                slp (int): safe level of p
                sln (int): safe level of n
                n_label (int): label of n

            Returns:
                float: gap
            """
            delta = 0
            if sln == 0 and slp > 0:
                return delta
            else:
                sl_ratio = slp/sln
                if sl_ratio == 1:
                    delta = self.random_state.random_sample()
                elif sl_ratio > 1:
                    delta = self.random_state.random_sample()/sl_ratio
                else:
                    delta = 1.0 - self.random_state.random_sample()*sl_ratio
            if not n_label == self.min_label:
                delta = delta*sln/(n_neighbors)
            return delta

        # generating samples
        trials = 0
        samples = []
        while len(samples) < n_to_sample:
            p_idx = self.random_state.choice(minority_indices)
            # extract random neighbor of p
            n_idx = self.random_state.choice(indices[p_idx][1:-1])

            # checking can-create criteria
            slp = safe_level(p_idx)
            sln = safe_level(p_idx, n_idx)

            if (not slp == 0) or (not sln == 0):
                # can create
                p = X[p_idx]
                n = X[n_idx]
                x_new = p.copy()

                for a in range(d):
                    delta = random_gap(slp, sln, y[n_idx])
                    diff = n[a] - p[a]
                    x_new[a] = p[a] + delta*diff
                samples.append(x_new)

            trials = trials + 1
            if len(samples)/trials < 1.0/n_to_sample:
                _logger.info(self.__class__.__name__ + ": " +
                             "no instances with slp > 0 and sln > 0 found")
                return X.copy(), y.copy()

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
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
