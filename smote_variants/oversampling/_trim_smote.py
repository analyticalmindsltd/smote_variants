"""
This module implements the TRIM_SMOTE method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['TRIM_SMOTE']

class TRIM_SMOTE(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 min_precision=0.3,
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
            n_neighbors (int): control parameter of the nearest neighbor component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            min_precision (float): minimum value of the precision
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(min_precision, 'min_precision', [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.min_precision = min_precision
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
            X (np.array): a subset of the training data
            y (np.array): an array of target labels
            split_on_border (bool): wether splitting on class borders is
                                    considered

        Returns:
            tuple(int, float), bool: (splitting feature, splitting value),
                                        make the split
        """
        trim_value = self.trim(y)
        max_t_minus_gain = 0.0
        split = None

        # checking all dimensions of X
        for idx in range(X.shape[1]):
            # sort the elements in dimension i
            sorted_X_y = sorted(zip(X[:, idx], y), key=lambda pair: pair[0]) # pylint: disable=invalid-name
            sorted_y = [yy for _, yy in sorted_X_y]

            # number of minority samples on the left
            left_min = 0
            # number of minority samples on the right
            right_min = np.sum(sorted_y == self.min_label)

            # check all possible splitting points sequentially
            for jdx in range(0, len(sorted_y)-1):
                if sorted_y[jdx] == self.min_label:
                    # adjusting the number of minority and majority samples
                    left_min = left_min + 1
                    right_min = right_min - 1
                # checking if we can split on the border and do not split
                # tie-ing feature values
                if ((split_on_border is False
                     or (split_on_border is True
                        and not sorted_y[jdx-1] == sorted_y[jdx]))
                        and sorted_X_y[jdx][0] != sorted_X_y[jdx+1][0]):
                    # compute trim value of the left
                    trim_left = left_min**2/(jdx+1)
                    # compute trim value of the right
                    trim_right = right_min**2/(len(sorted_y) - jdx - 1)
                    # let's check the gain
                    if max([trim_left, trim_right]) > max_t_minus_gain:
                        max_t_minus_gain = max([trim_left, trim_right])
                        split = (idx, sorted_X_y[jdx][0])

        # return splitting values and the value of the logical condition
        # in line 9
        if split is not None:
            return split, max_t_minus_gain > trim_value

        return (0, 0), False

    def leaf_loop(self, leafs, candidates):
        """
        The leaf loop from the paper.

        Args:
            leafs (list): the actual leafs
            candidates (list): the candidates

        Returns:
            list, list: the candidates and the potential leafs
        """
        add_to_leafs = []

        # small leafs are thrown away as noise
        leafs = [leaf for leaf in leafs if len(leaf[0]) > 1]

        # executing the loop starting in line 3
        for leaf in leafs:
            # the function implements the loop starting in line 6
            # splitting on class border is forced
            split, gain = self.determine_splitting_point(leaf[0],
                                                         leaf[1],
                                                         True)
            # condition in line 9
            if gain:
                # making the split
                mask_left = (leaf[0][:, split[0]] <= split[1])
                X_left = leaf[0][mask_left] # pylint: disable=invalid-name
                y_left = leaf[1][mask_left]
                mask_right = np.logical_not(mask_left)
                X_right = leaf[0][mask_right] # pylint: disable=invalid-name
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

        return candidates, add_to_leafs

    def candidate_loop(self, leafs, candidates, seeds):
        """
        The candidate loop from the paper.

        Args:
            leafs (list): the actual leafs
            candidates (list): the candidates
            seeds (list): the actual seeds

        Returns:
            list, list: the updated leafs and seeds lists
        """
        # small candidates are thrown away as noise
        candidates = [cand for cand in candidates if len(cand[0]) > 1]

        # iterating through all candidates (loop starting in line 21)
        for cand in candidates:
            # extracting splitting points, this time split on border
            # is not forced
            split, gain = self.determine_splitting_point(cand[0], cand[1], False)

            # checking condition in line 27
            if gain:
                # doing the split
                mask_left = (cand[0][:, split[0]] <= split[1])
                X_left, y_left = cand[0][mask_left], cand[1][mask_left] # pylint: disable=invalid-name
                mask_right = ~mask_left
                X_right, y_right = cand[0][mask_right], cand[1][mask_right] # pylint: disable=invalid-name
                # checking logic in line 29
                if np.sum(y_left == self.min_label) > 0:
                    leafs.append((X_left, y_left))
                # checking logic in line 31
                if np.sum(y_right == self.min_label) > 0:
                    leafs.append((X_right, y_right))
            else:
                # adding candidate to seeds (line 35)
                seeds.append(cand)

        return leafs, seeds

    def trimming(self, X, y):
        """
        Do the trimming.

        Args:
            X (np.array): the training vectors
            y (np.array): the target labels

        Returns:
            list: the seeds
        """
        leafs = [(X, y)]
        candidates = []
        seeds = []

        # executing the trimming
        # loop in line 2 of the paper
        _logger.info("%s: do the trimming process",
                        self.__class__.__name__)

        while len(leafs) > 0 or len(candidates) > 0:
            candidates, add_to_leafs = self.leaf_loop(leafs, candidates)

            # we implement line 15 and 18 by replacing the list of leafs by
            # the list of new leafs.
            leafs = add_to_leafs

            leafs, seeds = self.candidate_loop(leafs, candidates, seeds)

            # line 33 and line 36 are implemented by emptying the candidates
            # list
            candidates = []

        return seeds

    def generate_samples(self, X, y,
                            X_seed_min, # pylint: disable=invalid-name
                            n_to_sample):
        """
        Generate samples.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            X_seed_min (np.array): the seed points
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        n_neighbors = min([len(X_seed_min), self.n_neighbors+1])

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        nnmt= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X_seed_min)
        indices = nnmt.kneighbors(X_seed_min, return_distance=False)

        #n_dim_orig = self.n_dim
        #self.n_dim = np.min([self.n_dim, X_seed_min.shape[0]])
        samples = self.sample_simplex(X=X_seed_min,
                                      indices=indices,
                                      n_to_sample=n_to_sample)
        #self.n_dim = n_dim_orig

        return samples

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

        seeds = self.trimming(X, y)

        # filtering the resulting set
        filtered_seeds = [s for s in seeds if self.precision(s[1]) > self.min_precision]

        # fix for bad choice of min_precision
        multiplier = 0.9
        while len(filtered_seeds) == 0:
            threshold = self.min_precision * multiplier
            filtered_seeds = [s for s in seeds
                                if self.precision(s[1]) > threshold]
            multiplier = multiplier * 0.9

        seeds = filtered_seeds

        X_seed = np.vstack([s[0] for s in seeds]) # pylint: disable=invalid-name
        y_seed = np.hstack([s[1] for s in seeds])

        _logger.info("%s: do the sampling", self.__class__.__name__)
        # generating samples by SMOTE
        X_seed_min = X_seed[y_seed == self.min_label] # pylint: disable=invalid-name

        samples = self.generate_samples(X, y, X_seed_min, n_to_sample)

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
                'min_precision': self.min_precision,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
