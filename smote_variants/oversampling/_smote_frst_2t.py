"""
This module implements the SMOTE_FRST_2T method.
"""

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

from ..base import coalesce, coalesce_dict
from ..base import OverSampling

from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOTE_FRST_2T']

class SMOTE_FRST_2T(OverSampling):
    """
    References:
        * BibTex::

            @article{smote_frst_2t,
                        title = "Fuzzy-rough imbalanced learning for the
                                    diagnosis of High Voltage Circuit
                                    Breaker maintenance: The SMOTE-FRST-2T
                                    algorithm",
                        journal = "Engineering Applications of Artificial
                        Intelligence",
                        volume = "48",
                        pages = "134 - 139",
                        year = "2016",
                        issn = "0952-1976",
                        doi = "https://doi.org/10.1016/j.engappai.2015.10.009",
                        author = "Ramentol, E. and Gondres, I. and Lajes, S.
                                    and Bello, R. and Caballero,Y. and
                                    Cornelis, C. and Herrera, F.",
                        keywords = "High Voltage Circuit Breaker (HVCB),
                                    Imbalanced learning, Fuzzy rough set
                                    theory, Resampling methods"
                        }

    Notes:
        * Unlucky setting of parameters might result 0 points added, we have
            fixed this by increasing the gamma_S threshold if the number of
            samples accepted is low.
        * Similarly, unlucky setting of parameters might result all majority
            samples turned into minority.
        * In my opinion, in the algorithm presented in the paper the
            relations are incorrect. The authors talk about accepting samples
            having POS score below a threshold, and in the algorithm in
            both places POS >= gamma is used.
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_application,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 gamma_S=0.7, # pylint: disable=invalid-name
                 gamma_M=0.03, # pylint: disable=invalid-name
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors in the SMOTE sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            gamma_S (float): threshold of synthesized samples
            gamma_M (float): threshold of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(gamma_S, "gamma_S", 0)
        self.check_greater_or_equal(gamma_M, "gamma_M", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.gamma_S = gamma_S # pylint: disable=invalid-name
        self.gamma_M = gamma_M # pylint: disable=invalid-name
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, ss_params_default)
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'gamma_S': [0.8, 1.0],
                                  'gamma_M': [0.03, 0.05, 0.1]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def pos_cache(self,
                    X_0, # pylint: disable=invalid-name
                    X_1 # pylint: disable=invalid-name
                    ):
        """
        Calculate the matrix used to derive the positive membership values

        Args:
            X_0 (np.array): one array of vectors
            X_1 (np.array): another array of vectors

        Returns:
            np.array: the matrix that can be used to calculate the
                        membership values of X_1
        """
        # because of the minmax scaling, the clipping in eq. 1 is not needed
        dist = pairwise_distances(X_0, X_1, metric='l1')
        R_xy = X_0.shape[1] - dist # pylint: disable=invalid-name
        # taking the Lukasiewitcz T-norm
        R_xy = R_xy - X_0.shape[1] + 1.0 # pylint: disable=invalid-name
        R_xy[R_xy < 0.0] = 0.0

        pos = 1.0 - R_xy

        return pos

    def iteration(self, X, y, X_min, smote):
        """
        The function implementing the main iteration of the algorithm

        Args:
            X (np.array): the training vectors
            y (np.array): the target labels
            X_min (np.array): the minority samples
            smote (obj): the smote object used for sampling

        Returns:
            np.array, np.array: the updated majority and new minority samples
        """
        X_maj = X[y == self.maj_label]

        pos_cache = self.pos_cache(X_min, X_maj)

        # initializing some lists containing the results
        result_synth = np.zeros(shape=(0, X.shape[1]))
        result_maj = np.zeros(shape=(0, X.shape[1]))
        iteration = 0

        gammas = (self.gamma_S, self.gamma_M)

        while ((len(X_min) + len(result_synth) + len(result_maj)) < len(X_maj)
                    and iteration < 100):
            _logger.info("%s: iteration: %d",
                            self.__class__.__name__, iteration)

            # checking if the parameters aren't too conservative
            if len(result_synth) < iteration:
                gammas = (gammas[0] * 1.1, gammas[1])
                _logger.info("%s: gamma_S increased to %f",
                            self.__class__.__name__, gammas[0])

            X_samp = smote.sample(X, y)[0][X.shape[0]:]

            # computing POS membership values for the new samples
            pos_synth = self.pos_cache(X_min, X_samp)

            # adding samples with POS membership smaller than gamma_S to the
            # minority set
            to_add = np.where(np.min(pos_synth, axis=0) < gammas[0])[0]
            result_synth = np.vstack([result_synth, X_samp[to_add]])

            # checking the minimum POS values of the majority samples
            to_remove = np.where(np.min(pos_cache, axis=0) < gammas[1])[0]

            # if the number of majority samples with POS membership smaller
            # than gamma_M is not extreme, then changing labels, otherwise
            # decreasing gamma_M
            if len(to_remove) > (len(X_maj) - len(X_min))/2:
                to_remove = np.array([])
                gammas = (gammas[0], gammas[1] * 0.9)
                _logger.info("%s: gamma_M decreased to %f",
                                self.__class__.__name__, gammas[1])
            elif len(to_remove) > 0:
                result_maj = np.vstack([result_maj, X_maj[to_remove]])
                X_maj = np.delete(X_maj, to_remove, axis=0)
                pos_cache = np.delete(pos_cache, to_remove, axis=1)
                pos_cache = np.vstack([pos_cache,
                                self.pos_cache(result_maj[-len(to_remove):],
                                                X_maj)])

            # updating pos cache
            if len(to_add) > 0:
                pos_cache = np.vstack([pos_cache,
                                self.pos_cache(X_samp[to_add],
                                                X_maj)])

            _logger.info("%s: minority added: %d, majority removed: %d",
                        self.__class__.__name__, len(to_add), len(to_remove))
            iteration = iteration + 1

        return X_maj, np.vstack([result_synth, result_maj])

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        if np.sum(y == self.maj_label) <= np.sum(y == self.min_label):
            return self.return_copies(X, y, "Sampling is not needed")

        # Turning the ranges to 1 speeds up the positive membership
        # calculations
        mmscaler = MinMaxScaler()
        X = mmscaler.fit_transform(X)

        X_min = X[y == self.min_label]

        # after MinMax scaling, the POS value can be calculated as follows
        # determine proportion
        diff = np.sum(y == self.maj_label) - np.sum(y == self.min_label)
        prop = np.max([1.1 / diff, 0.2])

        # executing SMOTE to generate some minority samples
        smote = SMOTE(proportion=prop,
                        n_neighbors=self.n_neighbors,
                        nn_params=self.nn_params,
                        ss_params=self.ss_params,
                        n_jobs=self.n_jobs,
                        random_state=self._random_state_init)

        # iterating until the dataset becomes balanced
        X_maj, result_min = self.iteration(X, y, X_min, smote)

        # packing the results
        X_res = np.vstack([X_maj, X_min, result_min]) # pylint: disable=invalid-name

        y_res_maj = np.repeat(self.maj_label, len(X_maj))
        y_res_min = np.repeat(self.min_label, len(X_res) - len(X_maj))
        y_res = np.hstack([y_res_maj, y_res_min])

        return mmscaler.inverse_transform(X_res), y_res

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'gamma_S': self.gamma_S,
                'gamma_M': self.gamma_M,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
