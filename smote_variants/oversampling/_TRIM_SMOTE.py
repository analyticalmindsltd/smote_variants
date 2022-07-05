import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['TRIM_SMOTE']

class TRIM_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{trim_smote,
                            author="Puntumapon, Kamthorn
                            and Waiyamai, Kitsana",
                            editor="Tan, Pang-Ning
                            and Chawla, Sanjay
                            and Ho, Chin Kuan
                            and Bailey, James",
                            title="A Pruning-Based Approach for Searching
                                    Precise and Generalized Region for
                                    Synthetic Minority Over-Sampling",
                            booktitle="Advances in Knowledge Discovery
                                        and Data Mining",
                            year="2012",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="371--382",
                            isbn="978-3-642-30220-6"
                            }

    Notes:
        * It is not described precisely how the filtered data is used for
            sample generation. The method is proposed to be a preprocessing
            step, and it states that it applies sample generation to each
            group extracted.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 min_precision=0.3,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): control parameter of the nearest neighbor component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            min_precision (float): minimum value of the precision
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(min_precision, 'min_precision', [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.min_precision = min_precision
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
                                  'min_precision': [0.3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def trim(self, y):
        """
        Determines the trim value.

        Args:
            y (np.array): array of target labels

        Returns:
            float: the trim value
        """
        return np.sum(y == self.min_label)**2/len(y)

    def precision(self, y):
        """
        Determines the precision value.

        Args:
            y (np.array): array of target labels

        Returns:
            float: the precision value
        """
        return np.sum(y == self.min_label)/len(y)

    def determine_splitting_point(self, X, y, split_on_border=False):
        """
        Determines the splitting point.

        Args:
            X (np.matrix): a subset of the training data
            y (np.array): an array of target labels
            split_on_border (bool): wether splitting on class borders is
                                    considered

        Returns:
            tuple(int, float), bool: (splitting feature, splitting value),
                                        make the split
        """
        trim_value = self.trim(y)
        d = len(X[0])
        max_t_minus_gain = 0.0
        split = None

        # checking all dimensions of X
        for i in range(d):
            # sort the elements in dimension i
            sorted_X_y = sorted(zip(X[:, i], y), key=lambda pair: pair[0])
            sorted_y = [yy for _, yy in sorted_X_y]

            # number of minority samples on the left
            left_min = 0
            # number of minority samples on the right
            right_min = np.sum(sorted_y == self.min_label)

            # check all possible splitting points sequentiall
            for j in range(0, len(sorted_y)-1):
                if sorted_y[j] == self.min_label:
                    # adjusting the number of minority and majority samples
                    left_min = left_min + 1
                    right_min = right_min - 1
                # checking of we can split on the border and do not split
                # tieing feature values
                if ((split_on_border is False
                     or (split_on_border is True
                         and not sorted_y[j-1] == sorted_y[j]))
                        and sorted_X_y[j][0] != sorted_X_y[j+1][0]):
                    # compute trim value of the left
                    trim_left = left_min**2/(j+1)
                    # compute trim value of the right
                    trim_right = right_min**2/(len(sorted_y) - j - 1)
                    # let's check the gain
                    if max([trim_left, trim_right]) > max_t_minus_gain:
                        max_t_minus_gain = max([trim_left, trim_right])
                        split = (i, sorted_X_y[j][0])
        # return splitting values and the value of the logical condition
        # in line 9
        if split is not None:
            return split, max_t_minus_gain > trim_value
        else:
            return (0, 0), False

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

        leafs = [(X, y)]
        candidates = []
        seeds = []

        # executing the trimming
        # loop in line 2 of the paper
        _logger.info(self.__class__.__name__ +
                     ": " + "do the trimming process")
        while len(leafs) > 0 or len(candidates) > 0:
            add_to_leafs = []
            # executing the loop starting in line 3
            for leaf in leafs:
                # the function implements the loop starting in line 6
                # splitting on class border is forced
                split, gain = self.determine_splitting_point(
                    leaf[0], leaf[1], True)
                if len(leaf[0]) == 1:
                    # small leafs with 1 element (no splitting point)
                    # are dropped as noise
                    continue
                else:
                    # condition in line 9
                    if gain:
                        # making the split
                        mask_left = (leaf[0][:, split[0]] <= split[1])
                        X_left = leaf[0][mask_left]
                        y_left = leaf[1][mask_left]
                        mask_right = np.logical_not(mask_left)
                        X_right = leaf[0][mask_right]
                        y_right = leaf[1][mask_right]

                        # condition in line 11
                        if np.sum(y_left == self.min_label) > 0:
                            add_to_leafs.append((X_left, y_left))
                        # condition in line 13
                        if np.sum(y_right == self.min_label) > 0:
                            add_to_leafs.append((X_right, y_right))
                    else:
                        # line 16
                        candidates.append(leaf)
            # we implement line 15 and 18 by replacing the list of leafs by
            # the list of new leafs.
            leafs = add_to_leafs

            # iterating through all candidates (loop starting in line 21)
            for c in candidates:
                # extracting splitting points, this time split on border
                # is not forced
                split, gain = self.determine_splitting_point(c[0], c[1], False)
                if len(c[0]) == 1:
                    # small leafs are dropped as noise
                    continue
                else:
                    # checking condition in line 27
                    if gain:
                        # doing the split
                        mask_left = (c[0][:, split[0]] <= split[1])
                        X_left, y_left = c[0][mask_left], c[1][mask_left]
                        mask_right = np.logical_not(mask_left)
                        X_right, y_right = c[0][mask_right], c[1][mask_right]
                        # checking logic in line 29
                        if np.sum(y_left == self.min_label) > 0:
                            leafs.append((X_left, y_left))
                        # checking logic in line 31
                        if np.sum(y_right == self.min_label) > 0:
                            leafs.append((X_right, y_right))
                    else:
                        # adding candidate to seeds (line 35)
                        seeds.append(c)
            # line 33 and line 36 are implemented by emptying the candidates
            # list
            candidates = []

        # filtering the resulting set
        filtered_seeds = [s for s in seeds if self.precision(
            s[1]) > self.min_precision]

        # handling the situation when no seeds were found
        if len(seeds) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "no seeds identified")
            return X.copy(), y.copy()

        # fix for bad choice of min_precision
        multiplier = 0.9
        while len(filtered_seeds) == 0:
            filtered_seeds = [s for s in seeds if self.precision(
                s[1]) > self.min_precision*multiplier]
            multiplier = multiplier*0.9
            if multiplier < 0.1:
                _logger.warning(self.__class__.__name__ + ": " +
                                "no clusters passing the filtering")
                return X.copy(), y.copy()

        seeds = filtered_seeds

        X_seed = np.vstack([s[0] for s in seeds])
        y_seed = np.hstack([s[1] for s in seeds])

        _logger.info(self.__class__.__name__ + ": " + "do the sampling")
        # generating samples by SMOTE
        X_seed_min = X_seed[y_seed == self.min_label]
        if len(X_seed_min) <= 1:
            _logger.warning(self.__class__.__name__ + ": " +
                            "X_seed_min contains less than 2 samples")
            return X.copy(), y.copy()

        n_neighbors = min([len(X_seed_min), self.n_neighbors+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_seed_min)
        indices = nn.kneighbors(X_seed_min, return_distance=False)

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.randint(len(X_seed_min))
            random_neighbor_idx = self.random_state.choice(
                indices[random_idx][1:])
            samples.append(self.sample_between_points(
                X_seed_min[random_idx], X_seed_min[random_neighbor_idx]))

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
                'min_precision': self.min_precision,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
