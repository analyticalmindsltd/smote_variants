"""
This module implements the OUPS method.
"""

import numpy as np

from sklearn.linear_model import LogisticRegression

from ..base import coalesce_dict
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['OUPS']

class OUPS(OverSamplingSimplex):
    """
    References:
        * BibTex::

            @article{oups,
                        title = "A priori synthetic over-sampling methods for
                                    increasing classification sensitivity in
                                    imbalanced data sets",
                        journal = "Expert Systems with Applications",
                        volume = "66",
                        pages = "124 - 135",
                        year = "2016",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2016.09.010",
                        author = "William A. Rivera and Petros Xanthopoulos",
                        keywords = "SMOTE, OUPS, Class imbalance,
                                    Classification"
                        }

    Notes:
        * In the description of the algorithm a fractional number p (j) is
            used to index a vector.
    """

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_sample_ordinary]

    def __init__(self,
                 proportion=1.0,
                 *,
                 ss_params=None,
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
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def determine_sorting(self, X, y):
        """
        Determine the sorting of vectors.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels

        Returns:
            np.array: the sorting
        """
        # extracting propensity scores
        logreg = LogisticRegression(solver='lbfgs',
                                n_jobs=self.n_jobs,
                                random_state=self._random_state_init)
        logreg.fit(X, y)
        propensity = logreg.predict_proba(X)
        propensity = propensity[:,
                    np.where(logreg.classes_ == self.min_label)[0][0]]

        sorting = propensity.argsort()[::-1]

        return sorting

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

        sorting = self.determine_sorting(X, y)

        X_sorted = X[sorting] # pylint: disable=invalid-name

        min_indices = y[sorting] == self.min_label
        X_sorted_min = X_sorted[min_indices] # pylint: disable=invalid-name
        p = np.sum(y == self.maj_label) / np.sum(y == self.min_label) # pylint: disable=invalid-name
        simplex_weights = np.tile(np.arange(p, 0, -1.0), (X.shape[0], 1))
        simplex_weights = np.clip(simplex_weights, 0.0, 1.0)

        indices = np.vstack([np.arange(0, len(X)) + idx
                            for idx in range(1, simplex_weights.shape[1] + 1)]).T

        simplex_weights[indices >= len(X)] = 0.0
        indices = indices[min_indices]

        mask = ~ np.all(indices >= X.shape[0], axis=1)

        indices = indices[mask]
        X_sorted_min = X_sorted_min[mask] # pylint: disable=invalid-name
        indices[indices >= X.shape[0]] = X.shape[0] - indices[indices >= X.shape[0]] - 1

        #n_dim_orig = self.n_dim
        #self.n_dim = np.min([self.n_dim, simplex_weights.shape[1] + 1])

        samples = self.sample_simplex(X=X_sorted_min,
                                        indices=indices,
                                        n_to_sample=n_to_sample,
                                        X_vertices=X_sorted,
                                        simplex_weights=simplex_weights)
        #self.n_dim = n_dim_orig

        # sorting indices according to propensity scores
        #prop_sorted = sorted(zip(propensity,
        #                np.arange(len(propensity))), key=lambda x: -x[0])
        #p = np.sum(y == self.maj_label) / np.sum(y == self.min_label)
        #n = 0
        #samples = []
        #
        ## implementing Algorithm 1 in the cited paper with some minor changes
        ## to enable the proper sampling of p numbers
        #while n < len(propensity) and len(samples) < n_to_sample:
        #    if (y[prop_sorted[n][1]] == self.min_label
        #            and n < len(propensity) - 1):
        #        num = 1
        #        p_tmp = p
        #        while p_tmp > 0 and n + num < len(propensity):
        #            if self.random_state.random_sample() < p_tmp:
        #                samples.append(self.sample_between_points(
        #                    X[prop_sorted[n][1]], X[prop_sorted[n+num][1]]))
        #            p_tmp = p_tmp - 1
        #            num = num + 1
        #    n = n + 1

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
