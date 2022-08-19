"""
This module implements the NEATER method.
"""

import numpy as np

from ..base import coalesce, coalesce_dict
from ..base._simplexsampling import array_array_index
from ..base import (NearestNeighborsWithMetricTensor,
                                pairwise_distances_mahalanobis)
from ..base import OverSampling
from ._smote import SMOTE
from ._adasyn import ADASYN

from .._logger import logger
_logger= logger

__all__= ['NEATER']

class NEATER(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{neater,
                            author={Almogahed, B. A. and Kakadiaris, I. A.},
                            booktitle={2014 22nd International Conference on
                                         Pattern Recognition},
                            title={NEATER: Filtering of Over-sampled Data
                                    Using Non-cooperative Game Theory},
                            year={2014},
                            volume={},
                            number={},
                            pages={1371-1376},
                            keywords={data handling;game theory;information
                                        filtering;NEATER;imbalanced data
                                        problem;synthetic data;filtering of
                                        over-sampled data using non-cooperative
                                        game theory;Games;Game theory;Vectors;
                                        Sociology;Statistics;Silicon;
                                        Mathematical model},
                            doi={10.1109/ICPR.2014.245},
                            ISSN={1051-4651},
                            month={Aug}}

    Notes:
        * Evolving both majority and minority probabilities as nothing ensures
            that the probabilities remain in the range [0,1], and they need to
            be normalized.
        * The inversely weighted function needs to be cut at some value (like
            the alpha level), otherwise it will overemphasize the utility of
            having differing neighbors next to each other.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 smote_n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 b=5,
                 alpha=0.1,
                 h=20,
                 n_jobs=1,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            smote_n_neighbors (int): number of neighbors in SMOTE sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): the simplex sampling parameters
            b (int): number of neighbors
            alpha (float): smoothing term
            h (int): number of iterations in evolution
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(smote_n_neighbors, "smote_n_neighbors", 1)
        self.check_greater_or_equal(b, "b", 1)
        self.check_greater_or_equal(alpha, "alpha", 0)
        self.check_greater_or_equal(h, "h", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.smote_n_neighbors = smote_n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = ss_params
        self.params = {'b': b,
                        'alpha': alpha,
                        'h': h}
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
                                  'smote_n_neighbors': [3, 5, 7],
                                  'b': [3, 5, 7],
                                  'alpha': [0.1],
                                  'h': [20]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    #def wprob_mixed(self, prob, i, indices, distm, offset):
    #    ind = indices[i][1:]
    #    term_0 = 1*prob[offset + i][0]*prob[ind, 0]
    #    term_1 = distm[i, ind]*(prob[offset + i][1]*prob[ind, 0] +
    #                            prob[offset + i][0]*prob[ind, 1])
    #    term_2 = 1*prob[offset + i][1]*prob[ind, 1]
    #    print(term_1.shape, term_2.shape)
    #    return np.sum(term_0 + term_1 + term_2)

    def wprob_mixed_vectorized(self, prob, indices, distm, offset):
        """
        Compute the mixed terms

        Args:
            prob (np.array): probabilities
            indices (np.array): neighborhood structure
            distm (np.array): distance matrix
            offset (int): offsetting the indices

        Returns:
            np.array: the mixed terms
        """
        ind = indices[:, 1:]
        term_0 = 1 * prob[offset:, 0][:, None] * prob[ind][:, :, 0]
        term_1_b = (prob[offset:, 1] * prob[ind][:, :, 0].T +
                    prob[offset:, 0] * prob[ind][:, :, 1].T)
        term_1 = array_array_index(distm, ind) * term_1_b.T
        term_2 = 1 * prob[offset:, 1][:, None] * prob[ind][:, :, 1]
        return np.sum(term_0 + term_1 + term_2, axis=1)

    #def wprob_min(self, prob, i, indices, distm):
    #    term_0 = 0*prob[indices[i][1:], 0]
    #    term_1 = distm[i, indices[i][1:]]*(1*prob[indices[i][1:], 0] +
    #                                    0*prob[indices[i][1:], 1])
    #    term_2 = 1*prob[indices[i][1:], 1]
    #    return np.sum(term_0 + term_1 + term_2)

    def wprob_min_vectorized(self, prob, indices, distm):
        """
        Compute the minority terms

        Args:
            prob (np.array): probabilities
            indices (np.array): neighborhood structure
            distm (np.array): distance matrix

        Returns:
            np.array: the minority terms
        """
        ind = indices[:, 1:]
        term_0 = 0*prob[ind, 0]
        term_1 = (array_array_index(distm, ind)\
                    * (1 * prob[ind][:, :, 0] + \
                       0 * prob[ind][:, :, 1]))
        term_2 = 1*prob[ind][:, :, 1]

        return np.sum(term_0 + term_1 + term_2, axis=1)

    #def wprob_maj(self, prob, i, indices, distm):
    #    term_0 = 1*prob[indices[i][1:], 0]
    #    term_1 = distm[i, indices[i][1:]]*(0*prob[indices[i][1:], 0] +
    #                                    1*prob[indices[i][1:], 1])
    #    term_2 = 0*prob[indices[i][1:], 1]
    #    return np.sum(term_0 + term_1 + term_2)

    def wprob_maj_vectorized(self, prob, indices, distm):
        """
        Compute the majority terms

        Args:
            prob (np.array): probabilities
            indices (np.array): neighborhood structure
            distm (np.array): distance matrix

        Returns:
            np.array: the minority terms
        """
        ind = indices[:, 1:]
        term_0 = 1*prob[ind, 0]
        term_1 = array_array_index(distm, ind)\
                        *(0 * prob[ind][:, :, 0] +\
                          1 * prob[ind][:, :, 1])
        term_2 = 0*prob[ind][:, :, 1]

        return np.sum(term_0 + term_1 + term_2, axis=1)

    def utilities(self, prob, X, indices, distm):
        """
        Computes the utilit function

        Args:
            prob (np.array): strategy probabilities
            X (np.array): the training vectors
            indices (np.array): the neighborhood structure
            distm (np.array): the distance matrix

        Returns:
            np.array, np.array, np.array: utility values, minority
                                            utilities, majority
                                            utilities
        """

        #domain = range(len(X_syn))
        offset = len(X)
        #util_mixed = np.array([self.wprob_mixed(prob, i, indices, distm,
        #                                       offset=offset) for i in domain])
        util_mixed = self.wprob_mixed_vectorized(prob, indices, distm, offset)
        util_mixed = np.hstack([np.repeat(0, X.shape[0]), util_mixed])

        #util_min = np.array([self.wprob_min(prob, i, indices, distm) for i in domain])
        util_min = self.wprob_min_vectorized(prob, indices, distm)
        util_min = np.hstack([np.repeat(0, X.shape[0]), util_min])

        #util_maj = np.array([self.wprob_maj(prob, i, indices, distm) for i in domain])
        util_maj = self.wprob_maj_vectorized(prob, indices, distm)
        util_maj = np.hstack([np.repeat(0, X.shape[0]), util_maj])

        return util_mixed, util_min, util_maj

    def evolution(self,
                    *,
                    prob,
                    X,
                    X_syn, # pylint: disable=invalid-name
                    indices,
                    distm,
                    alpha=None):
        """
        Executing one step of the probabilistic evolution

        Args:
            prob (np.array): strategy probabilities
            X (np.array): original vectors
            X_syn (np.array): new samples
            indices (np.array): the neighborhood structure
            distm (np.array): the distance matrix
            alpha (float): smoothing function

        Returns:
            np.array: updated probabilities
        """
        # binary indicator indicating synthetic instances
        synthetic = np.hstack([np.array([False]*len(X)),
                                np.array([True]*len(X_syn))])

        alpha = coalesce(alpha, self.params['alpha'])

        util_mixed, util_min, util_maj = self.utilities(prob, X,
                                                        indices, distm)

        prob_new = prob.copy()
        synthetic_values = prob[:, 1] * \
            (alpha + util_min)/(alpha + util_mixed)
        prob_new[:, 1] = np.where(synthetic, synthetic_values, prob[:, 1])

        synthetic_values = prob[:, 0] * \
            (alpha + util_maj)/(alpha + util_mixed)
        prob_new[:, 0] = np.where(synthetic, synthetic_values, prob[:, 0])

        norm_factor = np.sum(prob_new, axis=1)

        prob_new[:, 0] = prob_new[:, 0]/norm_factor
        prob_new[:, 1] = prob_new[:, 1]/norm_factor

        return prob_new

    def generate_new_samples(self, X, y, nn_params):
        """
        Generate new samples.

        Args:
            X (np.array): training vectors
            y (np.array): the target labels
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array: all vectors and targets
        """
        # Applying SMOTE and ADASYN
        X_0, y_0 = SMOTE(proportion=self.proportion, # pylint: disable=invalid-name
                         n_neighbors=self.smote_n_neighbors,
                         nn_params=nn_params,
                         ss_params=self.ss_params,
                         n_jobs=self.n_jobs,
                         random_state=self._random_state_init).sample(X, y)

        X_1, y_1 = ADASYN(n_neighbors=self.params['b'], # pylint: disable=invalid-name
                          nn_params=nn_params,
                          ss_params=self.ss_params,
                          n_jobs=self.n_jobs,
                          random_state=self._random_state_init).sample(X, y)

        X_new = np.vstack([X_0, X_1[len(X):]])
        y_new = np.hstack([y_0, y_1[len(y):]])

        return X_new, y_new

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
            return self.return_copies(X, y, "Sampling is not needed.")

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        X_all, y_all = self.generate_new_samples(X, y, nn_params)

        X_syn = X_all[len(X):] # pylint: disable=invalid-name

        # initializing strategy probabilities
        prob = np.zeros(shape=(len(X_all), 2))
        prob.fill(0.5)
        for idx in range(len(X)):
            if y[idx] == self.min_label:
                prob[idx, 0], prob[idx, 1] = 0.0, 1.0
            else:
                prob[idx, 0], prob[idx, 1] = 1.0, 0.0

        # Finding nearest neighbors, +1 as X_syn is part of X_all and nearest
        # neighbors will be themselves
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=self.params['b'] + 1,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X_all)
        indices = nnmt.kneighbors(X_syn, return_distance=False)

        # computing distances
        distm = pairwise_distances_mahalanobis(X_syn,
                                            Y=X_all,
                                            tensor=nn_params.get('metric_tensor', None))
        distm[distm == 0] = 1e-8
        distm = 1.0/distm
        distm[distm > self.params['alpha']] = self.params['alpha']

        # executing the evolution
        for _ in range(self.params['h']):
            prob = self.evolution(prob=prob,
                                    X=X,
                                    X_syn=X_syn,
                                    indices=indices,
                                    distm=distm)

        # determining final labels
        y_all[len(X):] = np.argmax(prob[len(X):], axis=1)

        return X_all, y_all

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'smote_n_neighbors': self.smote_n_neighbors,
                'b': self.params['b'],
                'alpha': self.params['alpha'],
                'h': self.params['h'],
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
