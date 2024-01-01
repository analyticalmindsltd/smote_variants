"""
This module implements the oversampling driven undersampling
"""

import numpy as np

from ..base import UnderSampling, instantiate_obj


class OversamplingDrivenUndersampling(UnderSampling):
    """
    The oversampling driven undersampling
    """

    def __init__(self, oversampler_specification, mode="random", random_state=None):
        """
        The constructor of the oversampling driven undersampling

        Args:
            oversampler_specification (tuple): the specification of the oversampler
            random_state (None|int|np.random.RandomState): the random seed or state to be used
            mode (str): 'random'/'farthest' - the mode of sample removal
        """
        UnderSampling.__init__(self, random_state=random_state)
        self.oversampler = instantiate_obj(oversampler_specification)
        self.mode = mode

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """

        parameter_combinations = {
            "oversampler_specification": [
                ("smote_variants", "SMOTE", {"random_state": 5}),
                ("smote_variants", "ADASYN", {"random_state": 5}),
                ("smote_variants", "Borderline_SMOTE1", {"random_state": 5}),
            ],
            "mode": ["random", "farthest"],
        }

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y):
        """
        Carry out the undersampling

        Args:
            X (np.array): the feature vectors
            y (np.array): the corresponding class labels

        Returns:
            np.array, np.array: the undersampled feature vectors and class labels
        """
        n_maj, n_min = np.bincount(y)

        X_samp, _ = self.oversampler.sample(X, y)
        X_new = X_samp[X.shape[0]:]
        X_min = X[y == 1]
        X_maj = X[y == 0]

        dists = X_new - X_maj[:, None]

        dists = np.min(np.sqrt(np.sum((dists) ** 2, axis=2)), axis=1)

        if self.mode == "random":
            dists = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
            inv_dists = 1.0 - dists

            p = inv_dists / np.sum(inv_dists)

            mask = self.random_state.choice(np.arange(n_maj), n_min, p=p, replace=False)
            X_maj = X_maj[mask]
        elif self.mode == "farthest":
            sorting = np.argsort(dists)
            X_maj = X_maj[sorting][:n_min]

        X_res = np.vstack([X_maj, X_min]).copy()  # pylint: disable=invalid-name
        y_res = np.hstack([np.repeat(0, X_maj.shape[0]), np.repeat(1, X_min.shape[0])])

        return X_res, y_res
