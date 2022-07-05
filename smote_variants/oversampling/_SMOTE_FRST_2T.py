import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances

from .._metric_tensor import MetricTensor
from ._OverSampling import OverSampling

from ._SMOTE import SMOTE

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
                 nn_params={},
                 gamma_S=0.7,
                 gamma_M=0.03,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors in the SMOTE sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            gamma_S (float): threshold of synthesized samples
            gamma_M (float): threshold of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(gamma_S, "gamma_S", 0)
        self.check_greater_or_equal(gamma_M, "gamma_M", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.gamma_S = gamma_S
        self.gamma_M = gamma_M
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
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'gamma_S': [0.8, 1.0],
                                  'gamma_M': [0.03, 0.05, 0.1]}
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

        # Turning the ranges to 1 speeds up the positive membership
        # calculations
        mmscaler = MinMaxScaler()
        X = mmscaler.fit_transform(X)

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        # extracting the attribute ranges

        d = len(X[0])

        # after MinMax scaling, the POS value can be calculated as follows
        pos_cache = pairwise_distances(X_min, X_maj, metric='l1')
        pos_cache = 1.0 - pos_cache
        pos_cache = pos_cache.clip(0, d)
        pos_cache = 1.0 - pos_cache

        # initializing some lists containing the results
        result_synth = []
        result_maj = []
        iteration = 0

        gamma_S = self.gamma_S
        gamma_M = self.gamma_M

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # iterating until the dataset becomes balanced
        while (len(X_min) + len(result_synth) + len(result_maj)) < len(X_maj):
            _logger.info(self.__class__.__name__ + ":" +
                         ("iteration: %d" % iteration))
            # checking if the parameters aren't too conservative
            if len(result_synth) < iteration:
                gamma_S = gamma_S*1.1
                _logger.info(self.__class__.__name__ + ": " +
                             "gamma_S increased to %f" % gamma_S)

            # determine proportion
            diff = (sum(y == self.maj_label) -
                    sum(y == self.min_label))
            prop = max(1.1/diff, 0.2)

            # executing SMOTE to generate some minority samples
            smote = SMOTE(proportion=prop,
                          n_neighbors=self.n_neighbors,
                          nn_params=nn_params,
                          n_jobs=self.n_jobs,
                          random_state=self._random_state_init)
            X_samp, y_samp = smote.sample(X, y)
            X_samp = X_samp[len(X):]

            new_synth = []

            # computing POS membership values for the new samples
            pos_synth = pairwise_distances(X_min, X_samp, metric='l1')
            pos_synth = 1.0 - pos_synth
            pos_synth = pos_synth.clip(0, d)
            pos_synth = 1.0 - pos_synth

            # adding samples with POS membership smaller than gamma_S to the
            # minority set
            min_pos = np.min(pos_synth, axis=0)
            to_add = np.where(min_pos < gamma_S)[0]
            result_synth.extend(X_samp[to_add])
            new_synth.extend(X_samp[to_add])

            # checking the minimum POS values of the majority samples
            min_pos = np.min(pos_cache, axis=0)
            to_remove = np.where(min_pos < self.gamma_M)[0]

            # if the number of majority samples with POS membership smaller
            # than gamma_M is not extreme, then changing labels, otherwise
            # decreasing gamma_M
            if len(to_remove) > (len(X_maj) - len(X_min))/2:
                to_remove = np.array([])
                gamma_M = gamma_M*0.9
                _logger.info(self.__class__.__name__ + ": " +
                             "gamma_M decreased to %f" % gamma_M)
            else:
                result_maj.extend(X_maj[to_remove])
                X_maj = np.delete(X_maj, to_remove, axis=0)
                pos_cache = np.delete(pos_cache, to_remove, axis=1)

            # updating pos cache
            if len(new_synth) > 0:
                pos_cache_new = pairwise_distances(
                    np.vstack(new_synth), X_maj, metric='l1')
                pos_cache_new = 1.0 - pos_cache_new
                pos_cache_new = pos_cache_new.clip(0, d)
                pos_cache_new = 1.0 - pos_cache_new

                pos_cache = np.vstack([pos_cache, pos_cache_new])

            message = "minority added: %d, majority removed %d"
            message = message % (len(to_add), len(to_remove))
            _logger.info(self.__class__.__name__ + ":" + message)

            iteration = iteration + 1

        # packing the results
        X_res = np.vstack([X_maj, X_min])
        if len(result_synth) > 0:
            X_res = np.vstack([X_res, np.vstack(result_synth)])
        if len(result_maj) > 0:
            X_res = np.vstack([X_res, np.vstack(result_maj)])

        if len(X_maj) == 0:
            _logger.warning('All majority samples removed')
            return mmscaler.inverse_transform(X), y

        y_res_maj = np.repeat(self.maj_label, len(X_maj))
        n_y_res_min = len(X_min) + len(result_synth) + len(result_maj)
        y_res_min = np.repeat(self.min_label, n_y_res_min)
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
                'random_state': self._random_state_init}

