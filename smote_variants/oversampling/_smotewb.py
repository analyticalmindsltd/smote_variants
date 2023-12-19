"""
This module implements the SMOTEWB method.
"""
import logging
import math

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from ..base import OverSampling, NearestNeighborsWithMetricTensor, coalesce
from ._smote import SMOTE

_logger = logging.getLogger("smote_variants")

__all__ = ["SMOTEWB"]


class SMOTEWB(OverSampling):
    """
    References:
        * BibTex::
           @article{SAGLAM2022117023,
                title = {A novel SMOTE-based resampling technique trough noise
                detection and the boosting procedure},
                journal = {Expert Systems with Applications},
                volume = {200},
                pages = {117023},
                year = {2022},
                issn = {0957-4174},
                doi = {https://doi.org/10.1016/j.eswa.2022.117023}}
    """

    categories = [
        OverSampling.cat_noise_removal,  # applies noise removal
        OverSampling.cat_uses_classifier,  # uses some advanced classifier
        OverSampling.cat_sample_ordinary,  # sampling is done in the SMOTE scheme
        OverSampling.cat_extensive,  # adds minority samples only
        OverSampling.cat_metric_learning,
    ]  # metric learning is applicable (uses nearest neighbors)

    sample_good = 1
    sample_lonely = 0
    sample_bad = -1

    def __init__(
        self,
        proportion=1.0,
        *,
        n_iters=100,
        max_depth=30,
        nn_params=None,
        n_jobs=1,
        random_state=None,
        **_kwargs,
    ):
        """
        Constructor of the SMOTE template object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_iters (int): number of iterations (number of weak classifiers, dtrees) of the ensemble
                                noise filtering process (in the paper: M)
            max_depth (int): maximum depth of dtrees (>=3)
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (None/int/np.random.RandomState): the random state
            _kwargs: for technical reasons and to facilitate serialization, additional
                     keyword arguments are accepted
        """

        OverSampling.__init__(self, random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_iters, "n_iters", 1)
        self.check_greater_or_equal(max_depth, "max_depth", 3)
        self.check_n_jobs(n_jobs, "n_jobs")

        self.proportion = proportion
        self.n_iters = n_iters
        self.max_depth = max_depth
        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.
        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {
            "proportion": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "n_iters": [30, 50, 100],
            "max_depth": [5, 10, 20, 30],
        }

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def boosted_weights(self, X, y):
        """
        Implementation of Algorithm 2.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.array): weight of the samples
        """
        n_samples = X.shape[0]
        w = np.repeat(1.0 / n_samples, n_samples)

        # based on boosted_weights.R
        clf = DecisionTreeClassifier(
            random_state=self._random_state_init,
            max_depth=self.max_depth,
            min_samples_split=3,
        )

        for _ in range(0, self.n_iters):
            clf.fit(X, y, sample_weight=w)
            predicted_labels = clf.predict(X)

            w_error = sum(w[predicted_labels != y])

            # no zero check in the original code (samples are not_noise)
            if w_error == 0:
                return np.zeros(n_samples, float)

            alpha = 0.5 * np.log((1 - w_error) / w_error)
            w[predicted_labels != y] = w[predicted_labels != y] * np.exp(alpha)

            w = w / sum(w)

        return w

    @staticmethod
    def noise_threshold(w, th):
        """
        Creating the noise mask

        Args:
            w (np.array): the weights
            th (float): the threshold

        Returns:
            np.array: the mask
        """
        noise = np.repeat(False, len(w))
        noise[w > th] = True
        return noise

    def noise_detection(self, X, y):
        """
        Implementation of Algorithm 3.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.array, np.array): Mask of noise for the minority and the majority
            samples, respectively.
        """

        w = self.boosted_weights(X, y)

        w_min = w[y == self.min_label]
        w_maj = w[y == self.maj_label]

        n_samples = len(y)

        # noise thresholds for majority and minority classes
        th_min = (2 * len(w_maj)) / (n_samples**2)
        th_maj = (2 * len(w_min)) / (n_samples**2)

        noise_mask_min = SMOTEWB.noise_threshold(w_min, th_min)
        noise_mask_maj = SMOTEWB.noise_threshold(w_maj, th_maj)

        return noise_mask_min, noise_mask_maj

    def first_majority_index(self, nearest_ys, no_majority_result=-1):
        """
        Finds the index of the first majority element in the nearest_ys per row.

        Args:
            nearest_ys (np.ndarray): label of the nearest neighbours
            no_majority_result (int): result if there is no majority item in a row
        Returns:
            np.array: the i-th element of the array is the index of the first majority label
                        in the i-th row of the nearest_ys, or no_majority_result if the row
                        contains no majority labels.
        """
        mask = nearest_ys == self.maj_label
        return np.where(mask.any(axis=1), mask.argmax(axis=1), no_majority_result)

    def sampling_algorithm(
        self, X, y
    ):  # pylint: disable=too-many-locals,too-many-statements
        """
        Does the sample generation according to the class parameters.

        TODO: break into smaller functions

        Args:
            X (np.ndarray): training set
            y (np.array): target labels
        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed.")

        # use logging in the below format
        _logger.info("%s: sampling", self.__class__.__name__)

        # Step 1: MinMaxScaler - Rescaling is ignored at the suggestion of the
        # author of the algorithm.

        # Step2: separating the classes
        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # Step 3-4: creating masks of noise samples
        noise_mask_min, noise_mask_maj = self.noise_detection(X, y)

        # Start: Algorithm 4
        n_min = len(X_min)
        n_maj = len(X_maj)

        # paper: floor, R implementation: ceil
        k_max = math.floor(n_maj / n_min)

        # non-noise samples and labels
        X_min_not_noise = X_min[~noise_mask_min]  # pylint: disable=invalid-name
        X_maj_not_noise = X_maj[~noise_mask_maj]  # pylint: disable=invalid-name
        X_not_noise = np.concatenate(  # pylint: disable=invalid-name
            [X_min_not_noise, X_maj_not_noise]
        )
        y_not_noise = np.concatenate(
            [
                np.repeat(self.min_label, X_min_not_noise.shape[0]),
                np.repeat(self.maj_label, X_maj_not_noise.shape[0]),
            ]
        )

        # fitting the model
        # paper: X^good_pos makes no sense, R code: x_notnoise
        nn_params = {**self.nn_params}
        nn_params["metric_tensor"] = self.metric_tensor_from_nn_params(nn_params, X, y)

        # all data points are noise
        if len(X_not_noise) == 0:
            smote = SMOTE(proportion=self.proportion, nn_params=self.nn_params)
            return smote.sample(X, y)

        n_neighbors = min([len(X_not_noise), k_max + 1])
        nnmt = NearestNeighborsWithMetricTensor(
            n_neighbors=n_neighbors, n_jobs=self.n_jobs, **nn_params
        )

        nnmt.fit(X_not_noise)
        indices = nnmt.kneighbors(X_min, n_neighbors=n_neighbors, return_distance=False)

        # number of the positive neighbors until the first negative one (per row)
        k_arr = self.first_majority_index(
            y_not_noise[indices], no_majority_result=k_max + 1
        )

        # decrease one if the sample is not a noise
        k_arr[~noise_mask_min] -= 1
        # in case there is a majority sample at 0 distance
        k_arr[k_arr == -1] = 0
        # all neighbours are positive
        k_max = min(indices.shape[1], k_max)
        k_arr[k_arr > k_max] = k_max

        # setting the labels of the samples of X_pos
        fl_arr = np.empty(n_min, int)
        fl_arr[k_arr > 0] = SMOTEWB.sample_good
        fl_arr[(k_arr == 0) & noise_mask_min] = SMOTEWB.sample_bad
        fl_arr[(k_arr == 0) & (~noise_mask_min)] = SMOTEWB.sample_lonely
        # End: Alg 4

        # Step 5: n_to_sample - done
        # Step 6: computing n_to_sample per samples
        n_min = len(X_min)

        # number of good and lonely samples
        n_good_min = np.sum(fl_arr == SMOTEWB.sample_good)
        n_lonely_min = np.sum(fl_arr == SMOTEWB.sample_lonely)

        # prevent errors (missing from the original) no good or lonely
        # samples => give chance to the others
        if n_good_min + n_lonely_min == 0:
            smote = SMOTE(proportion=self.proportion, nn_params=self.nn_params)
            return smote.sample(X, y)

        n_to_sample_per_sample = math.ceil(n_to_sample / (n_good_min + n_lonely_min))
        C = np.zeros(n_min, int)  # pylint: disable=invalid-name
        C[fl_arr == SMOTEWB.sample_good] = n_to_sample_per_sample
        C[fl_arr == SMOTEWB.sample_lonely] = n_to_sample_per_sample

        # Step 8: correcting the number of samples to be generated to achieve the desired balance
        # paper: ceil => n_to_sample-p.sum(C) zero or negative
        # we probably have too many samples because of the ceil
        diff = np.sum(C) - n_to_sample
        if diff > 0:
            good_and_lonely_ind = np.where(
                (fl_arr == SMOTEWB.sample_good) | (fl_arr == SMOTEWB.sample_lonely)
            )[0]
            selected_ind = self.random_state.choice(
                good_and_lonely_ind, diff, replace=len(good_and_lonely_ind) < diff
            )
            C[selected_ind] -= 1

        # Step 9: sample generation
        synt_sample_list = []
        for i in range(0, n_min):
            if fl_arr[i] == SMOTEWB.sample_lonely:
                for j in range(C[i]):
                    synt_sample_list.append(X_min[i].copy())
            elif fl_arr[i] == SMOTEWB.sample_good and C[i] > 0:
                nn = (
                    indices[i, 0 : k_arr[i]]
                    if noise_mask_min[i]
                    else indices[i, 1 : k_arr[i] + 1]
                )

                if len(nn) > 0:
                    k_ids = self.random_state.choice(nn, C[i])
                    for j in k_ids:
                        synt_sample_list.append(
                            self.sample_between_points(X_min[i], X_not_noise[j])
                        )

        # Step 11-12: merging the original samples and the synthetic ones
        X_synt_samples = np.array(synt_sample_list)  # pylint: disable=invalid-name
        X_samples = np.vstack([X, X_synt_samples])  # pylint: disable=invalid-name

        return (
            X_samples,
            np.hstack([y, np.hstack([self.min_label] * len(synt_sample_list))]),
        )

    def preprocessing_transform(self, X):
        """
        Scale the attributes of X into [0, 1] range.

        Args:
            X (np.array): features
        Returns:
            np.array: transformed features
        """
        return X

    def get_params(self, deep=False):
        """
        Returns the parameters of the object

        Returns:
            dict: the parameters of the current sampling object
        """
        return {
            "proportion": self.proportion,
            "max_depth": self.max_depth,
            "nn_params": self.nn_params,
            "n_iters": self.n_iters,
            **OverSampling.get_params(self, deep),
        }
