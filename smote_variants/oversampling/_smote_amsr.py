"""
This module implements the SMOTE_AMSR method.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from ..base import OverSampling

from .._logger import logger

_logger = logger

__all__ = ["SMOTE_AMSR"]


class SMOTE_AMSR(OverSampling):
    """
    Description:
        combing the attention mechanism of sparse regions and the
        interpolation technique of geometric topology to Generate
        New Minority Data.
    """

    categories = [OverSampling.cat_extensive]

    # @_deprecate_positional_args
    def __init__(self, proportion=1.0, *, topology="mesh", random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            topology (str): the topology to use ('bus'/'star'/'mesh')
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """

        super().__init__(random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0.0)
        self.check_isin(topology, "topology", ["star", "bus", "mesh"])

        self.proportion = proportion
        self.topology = topology

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.
        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {
            "proportion": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "topology": ["bus", "star", "mesh"],
        }

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def generate_star_steps_ge(
        self, mean_anomaly, scores, sample_indices, greater_equal_mask, max_min, X
    ):
        """
        The vectorized greater or equal case

        Args:
            mean_anomaly (float): the mean anomaly
            scores (np.array): the classification scores
            sample_indices (np.array): the sample indices
            greater_equal_mask (np.array): the greater or equal mask
            max_min (float): the range
            X (np.array): the feature vectors

        Returns:
            np.array: the steps
        """
        d_ge = mean_anomaly - scores[sample_indices][greater_equal_mask]

        if max_min > 0:
            d_ge = d_ge / max_min

        steps_ge = self.random_state.uniform(
            size=(np.sum(greater_equal_mask), X.shape[1])
        )
        steps_ge = (steps_ge.T * (1.0 - d_ge) + d_ge).T

        return steps_ge

    def generate_star_steps_less(
        self, scores, sample_indices, less_mask, mean_anomaly, max_min, X
    ):
        """
        The vectorized less case

        Args:
            scores (np.array): the classification scores
            sample_indices (np.array): the sample indices
            less_mask (np.array): the greater or equal mask
            mean_anomaly (float): the mean anomaly
            max_min (float): the range
            X (np.array): the feature vectors

        Returns:
            np.array: the steps
        """
        d_less = scores[sample_indices][less_mask] - mean_anomaly

        if max_min > 0:
            d_less = d_less / max_min

        steps_less = self.random_state.uniform(size=(np.sum(less_mask), X.shape[1]))
        steps_less = (steps_less.T * (1.0 - d_less)).T
        return steps_less

    def _star_generator(self, X, n_to_sample, scores, classifier):
        """
        Make samples with star topology

        Args:
            X (np.array): the feature vectory
            n_to_sample (int): the number of samples to generate
            scores (np.array): classification scores
            classifier (obj): the classifier

        Returns:
            np.array: the generated feature vectors
        """
        # samples = []

        X_mean = np.mean(X, axis=0)  # pylint: disable=invalid-name
        mean_anomaly = classifier.decision_function(X_mean.reshape(1, -1))[0]

        # this array stores all generated samples
        samples = np.zeros(shape=(n_to_sample, X.shape[1]))

        # the base points
        sample_indices = self.random_state.choice(X.shape[0], n_to_sample, replace=True)
        base_points = X[sample_indices]

        # differences
        diffs = X_mean - base_points

        # the anomaly greater or equal / less masks
        greater_equal_mask = mean_anomaly >= scores[sample_indices]
        less_mask = ~greater_equal_mask
        max_min = max(max(scores), mean_anomaly) - min(  # pylint: disable=nested-min-max
            min(scores), mean_anomaly
        )

        # the greater or equal case vectorized

        samples[greater_equal_mask] = (
            base_points[greater_equal_mask]
            + self.generate_star_steps_ge(
                mean_anomaly, scores, sample_indices, greater_equal_mask, max_min, X
            )
            * diffs[greater_equal_mask]
        )

        # the less case vectorized

        samples[less_mask] = (
            base_points[less_mask]
            + self.generate_star_steps_less(
                scores, sample_indices, less_mask, mean_anomaly, max_min, X
            )
            * diffs[less_mask]
        )

        return samples

    def generate_bus_steps_ge(self, diffs_scores_all, greater_equal_mask, max_min, X):
        """
        The greater or equal steps for the bus topology

        Args:
            diffs_scores_all (np.array): the differences
            greater_equal_mask (np.array): the mask
            max_min (float): the range
            X (np.array): the feature vectors

        Returns:
            np.array: the steps
        """
        d_ge = diffs_scores_all[greater_equal_mask]

        if max_min > 0:
            d_ge = diffs_scores_all[greater_equal_mask] / max_min

        steps_ge = self.random_state.uniform(
            size=(np.sum(greater_equal_mask), X.shape[1])
        )
        steps_ge = (steps_ge.T * (1.0 - d_ge) + d_ge).T

        return steps_ge

    def generate_bus_steps_less(self, diffs_scores_all, less_mask, max_min, X):
        """
        The less steps for the bus topology

        Args:
            diffs_scores_all (np.array): the differences
            less_mask (np.array): the mask
            max_min (float): the range
            X (np.array): the feature vectors

        Returns:
            np.array: the steps
        """
        d_less = -diffs_scores_all[less_mask]

        if max_min > 0:
            d_less = -diffs_scores_all[less_mask] / max_min

        steps_less = self.random_state.uniform(size=(np.sum(less_mask), X.shape[1]))
        steps_less = (steps_less.T * (1.0 - d_less)).T

        return steps_less

    def _bus_generator(self, X, n_to_sample, scores):
        """
        Make samples with bus topology

        Args:
            X (np.array): the feature vectory
            n_to_sample (int): the number of samples to generate
            scores (np.array): classification scores

        Returns:
            np.array: the generated feature vectors
        """

        max_min = max(scores) - min(scores)

        # this array stores all generated samples
        samples = np.zeros(shape=(n_to_sample, X.shape[1]))

        # the base points
        sample_indices = self.random_state.choice(
            X.shape[0] - 1, n_to_sample, replace=True
        )
        base_points = X[1:][sample_indices]
        diffs = X[:-1] - X[1:]
        diffs_all = diffs[sample_indices]
        diffs_scores = scores[:-1] - scores[1:]
        diffs_scores_all = diffs_scores[sample_indices]

        greater_equal_mask = diffs_scores_all >= 0
        less_mask = ~greater_equal_mask

        samples[greater_equal_mask] = (
            base_points[greater_equal_mask]
            + self.generate_bus_steps_ge(
                diffs_scores_all, greater_equal_mask, max_min, X
            )
            * diffs_all[greater_equal_mask]
        )

        samples[less_mask] = (
            base_points[less_mask]
            + self.generate_bus_steps_less(diffs_scores_all, less_mask, max_min, X)
            * diffs_all[less_mask]
        )

        return samples

    def generate_mesh_ge_steps(self, diffs_scores, greater_equal_mask, max_min, X):
        """
        The greater or equal steps for the mesh

        Args:
            diffs_scores (np.array): the differences
            greater_equal_mask (np.array): the mask
            max_min (float): the range
            X (np.array): the feature vectors

        Returns:
            np.array: the steps
        """
        d_ge = diffs_scores[greater_equal_mask]

        if max_min > 0:
            d_ge = diffs_scores[greater_equal_mask] / max_min

        steps_ge = self.random_state.uniform(
            size=(np.sum(greater_equal_mask), X.shape[1])
        )
        steps_ge = (steps_ge.T * (1.0 - d_ge) + d_ge).T
        return steps_ge

    def generate_mesh_less_steps(self, diffs_scores, less_mask, max_min, X):
        """
        The less steps for the mesh

        Args:
            diffs_scores (np.array): the differences
            less_mask (np.array): the mask
            max_min (float): the range
            X (np.array): the feature vectors

        Returns:
            np.array: the steps
        """
        d_less = -diffs_scores[less_mask]

        if max_min > 0:
            d_less = -diffs_scores[less_mask] / max_min

        steps_less = self.random_state.uniform(size=(np.sum(less_mask), X.shape[1]))
        steps_less = (steps_less.T * (1.0 - d_less)).T
        return steps_less

    def _mesh_generator(self, X, n_to_sample, scores):
        """
        Make samples with mesh topology

        Args:
            X (np.array): the feature vectory
            n_to_sample (int): the number of samples to generate
            scores (np.array): classification scores

        Returns:
            np.array: the generated feature vectors
        """

        # Implementation of the mesh topology
        max_min = max(scores) - min(scores)

        # this array stores all generated samples
        samples = np.zeros(shape=(n_to_sample, X.shape[1]))

        # the base points
        sample_indices = self.random_state.choice(X.shape[0], n_to_sample, replace=True)
        other_indices = self.random_state.choice(X.shape[0], n_to_sample, replace=True)
        base_points = X[sample_indices]
        other_points = X[other_indices]

        diffs = base_points - other_points
        diffs_scores = scores[sample_indices] - scores[other_indices]

        greater_equal_mask = diffs_scores >= 0
        less_mask = ~greater_equal_mask

        samples[greater_equal_mask] = (
            other_points[greater_equal_mask]
            + self.generate_mesh_ge_steps(diffs_scores, greater_equal_mask, max_min, X)
            * diffs[greater_equal_mask]
        )

        samples[less_mask] = (
            other_points[less_mask]
            + self.generate_mesh_less_steps(diffs_scores, less_mask, max_min, X)
            * diffs[less_mask]
        )

        return samples

    def _make_samples(self, X, n_to_sample, scores, classifier):
        """
        Make samples

        Args:
            X (np.array): the feature vectory
            n_to_sample (int): the number of samples to generate
            scores (np.array): classification scores
            classifier (obj): the classifier

        Returns:
            np.array: the generated feature vectors
        """
        samples = None
        if self.topology == "star":
            samples = self._star_generator(X, n_to_sample, scores, classifier)
        elif self.topology == "bus":
            samples = self._bus_generator(X, n_to_sample, scores)
        elif self.topology == "mesh":
            samples = self._mesh_generator(X, n_to_sample, scores)

        return samples

    def _in_danger_iforest(self, X, y, class_label):
        """
        Check if samples are indanger

        Args:
            X (np.array): feature vectors
            y (np.array): the target labels
            class_label (int): the class label to check

        Returns:
            np.array, clf: the class scores and the classifier
        """

        clf = IsolationForest(n_estimators=100, random_state=self._random_state_init)
        clf.fit(X)

        X_scores = clf.decision_function(X)  # pylint: disable=invalid-name
        class_scores = X_scores[y == class_label]

        return class_scores, clf

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        _logger.info(
            "%s: Running sampling via %s", self.__class__.__name__, self.descriptor()
        )

        X_min = X[y == self.min_label]

        n_to_sample = self.det_n_to_sample(
            self.proportion,
            self.class_stats[self.maj_label],
            self.class_stats[self.min_label],
        )
        if n_to_sample == 0:
            _logger.warning("%s: Sampling is not needed", self.__class__.__name__)
            return X, y

        mda_scores, clf_anomaly = self._in_danger_iforest(X, y, self.min_label)

        X_new = self._make_samples(X_min, n_to_sample, mda_scores, clf_anomaly)

        return np.vstack([X, X_new]), np.hstack(
            [y, np.repeat(self.min_label, X_new.shape[0])]
        )

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {
            "proportion": self.proportion,
            "topology": self.topology,
            "random_state": self._random_state_init,
        }
