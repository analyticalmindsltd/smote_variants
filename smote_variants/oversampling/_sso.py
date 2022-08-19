"""
This module implements the SSO method.
"""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

import scipy.special as sspecial

from ..base import (NearestNeighborsWithMetricTensor,
                                pairwise_distances_mahalanobis)
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SSO']

class SSO(OverSampling):
    """
    References:
        * BibTex::

            @InProceedings{sso,
                            author="Rong, Tongwen
                            and Gong, Huachang
                            and Ng, Wing W. Y.",
                            editor="Wang, Xizhao
                            and Pedrycz, Witold
                            and Chan, Patrick
                            and He, Qiang",
                            title="Stochastic Sensitivity Oversampling
                                    Technique for Imbalanced Data",
                            booktitle="Machine Learning and Cybernetics",
                            year="2014",
                            publisher="Springer Berlin Heidelberg",
                            address="Berlin, Heidelberg",
                            pages="161--171",
                            isbn="978-3-662-45652-1"
                            }

    Notes:
        * In the algorithm step 2d adds a constant to a vector. I have
            changed it to a componentwise adjustment, and also used the
            normalized STSM as I don't see any reason why it would be
            some reasonable, bounded value.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_classifier,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 h=10, # pylint: disable=invalid-name
                 n_iter=5,
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
            n_neighbors (int): number of neighbors
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            h (int): number of hidden units
            n_iter (int): optimization steps
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(h, "h", 1)
        self.check_greater_or_equal(n_iter, "n_iter", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.h = h # pylint: disable=invalid-name
        self.n_iter = n_iter
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
                                  'n_neighbors': [3, 5],
                                  'h': [2, 5, 10, 20],
                                  'n_iter': [5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    #def eq_6(self, Q, w, u, v, x, h):
    #    """
    #    Equation 6 in the paper
    #    """
    #    tmp_sum = np.zeros(h)
    #    for i in range(h):
    #        a = (x - u[i] + Q)/np.sqrt(2*v[i])
    #        b = (x - u[i] - Q)/np.sqrt(2*v[i])
    #        tmp_prod = (sspecial.erf(a) - sspecial.erf(b))
    #        tmp_sum[i] = np.sqrt(np.pi/2)*v[i]*np.prod(tmp_prod)
    #    return np.dot(tmp_sum, w)/(2*Q)**len(x)

    def eq_6_vectorized(self, *, Q, w, u, v, X): # pylint: disable=invalid-name
        """
        Vectorized implementation of Equation 6 in the paper

        Args:
            Q (np.array): the Q value
            w (np.array): the w vector
            u (np.array): the u vectors
            v (np.array): the v vector
            X (np.array): all training vectors

        Returns:
            np.array: eq. 6 for each X vector
        """
        tmp_a = (X[:, None] - u + Q) / np.sqrt(2 * v)[None, :, None]
        tmp_b = (X[:, None] - u - Q) / np.sqrt(2 * v)[None, :, None]
        tmp_prod = sspecial.erf(tmp_a) - sspecial.erf(tmp_b)
        tmp_sum = (np.sqrt(np.pi / 2.0) * v * np.prod(tmp_prod, axis=2))
        result = np.dot(tmp_sum, w) / (2 * Q) ** X.shape[1]
        return result

    #def eq_8(self, Q, w, u, v, x, h):
    #    """
    #    Equation 8 in the paper
    #    """
    #    res = 0.0
    #    for i in range(h):
    #        vi2 = v[i]**2
    #        for r in range(h):
    #            vr2 = v[r]**2
    #            a1 = (np.sqrt(2*vi2*vr2*(vi2 + vr2)))
    #
    #            a00_v = (vi2 + vr2)*(x + Q)
    #            a01_v = vi2*u[r] + vr2*u[i]
    #            a0_v = a00_v - a01_v
    #            a_v = a0_v/a1
    #
    #            b_v = ((vi2 + vr2)*(x - Q) - (vi2*u[r] + vr2*u[i]))/a1
    #            tmp_prod = sspecial.erf(a_v) - sspecial.erf(b_v)
    #
    #            tmp_a = (np.sqrt(2*vi2*vr2*(vi2 + vr2)) /
    #                        (vi2 + vr2))**len(x)
    #            norm = np.linalg.norm(u[r] - u[i])
    #            tmp_b = np.exp(-0.5 * norm**2/(vi2 + vr2))
    #
    #            res = res + tmp_a*tmp_b*np.prod(tmp_prod)*w[i]*w[r]
    #
    #    return (np.sqrt(np.pi)/(4*Q))**len(x)*res

    def eq_8_av_bv(self, *, Q, u, X, v2, vi_vr, a1):
        """
        Calculate the a_v and b_v arrays.

        Args:
            Q (np.array): the Q value
            u (np.array): the u vectors
            X (np.array): all training vectors
            v2 (np.array): the squared v values
            vi_vr (np.array): the cross sum of squared v values
            a1 (np.array): the precalculated a1 term

        Returns:
            np.array: eq. 8 for each X vector
        """

        a00_v = vi_vr[:, :, None, None] * (X + Q)

        tmp1 = np.repeat(v2, u.shape[0]).reshape(v2.shape[0], u.shape[0])[:, :, None]
        tmp2 = np.array([u] * u.shape[0])

        a01_v = u[:, None] * v2[:, None] + tmp2 * tmp1
        a0_v = a00_v - a01_v[:, :, None]
        a_v = a0_v / a1[:, :, None, None]

        b_v = vi_vr[:, :, None, None] * (X - Q)
        b_v = b_v - a01_v[:, :, None]
        b_v = b_v / a1[:, :, None, None]

        return a_v, b_v

    def eq_8_vectorized(self, *, Q, w, u, v, X): # pylint: disable=invalid-name
        """
        Vectorized implementation of Equation 8 in the paper

        Args:
            Q (np.array): the Q value
            w (np.array): the w vector
            u (np.array): the u vectors
            v (np.array): the v vector
            X (np.array): all training vectors

        Returns:
            np.array: eq. 8 for each X vector
        """
        v2 = v ** 2.0 # pylint: disable=invalid-name

        vi_vr = (v2[:, None] + v2)

        a1 = np.sqrt(2.0 * (v2[:, None] * v2) * vi_vr) # pylint: disable=invalid-name

        a_v, b_v = self.eq_8_av_bv(Q=Q, u=u, X=X, v2=v2, vi_vr=vi_vr, a1=a1)

        tmp_prod = sspecial.erf(a_v) - sspecial.erf(b_v)

        tmp_a = (a1 / vi_vr) ** X.shape[1]

        tmp_b = np.exp(-0.5 * np.linalg.norm(u[:, None] - u, axis=2)**2/vi_vr)

        result = (tmp_a * tmp_b * (w[:, None] * w))[:, :, None] \
                                        * np.prod(tmp_prod, axis=3)

        result = np.sum(np.sum(result, axis=0), axis=0)

        result = (np.sqrt(np.pi) / (4 * Q)) ** X.shape[1] * result

        return result

    def calculate_vectors(self, X, y, X_min, nn_params):
        """
        Calculate the vectors used in the rest of the algorithm.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            X_min (np.array): all minority samples
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array, np.array, np.array: the u, v, f and w
                                                    vectors
        """
        # applying kmeans clustering to find the hidden neurons
        h = min([self.h, len(X_min)]) # pylint: disable=invalid-name
        kmeans = KMeans(n_clusters=h,
                        random_state=self._random_state_init)
        kmeans.fit(X)

        # extracting the hidden center elements
        u = kmeans.cluster_centers_ # pylint: disable=invalid-name

        # extracting scale parameters as the distances of closest centers
        nn_cent = NearestNeighborsWithMetricTensor(n_neighbors=2,
                                                    n_jobs=self.n_jobs,
                                                    **nn_params)
        nn_cent.fit(u)
        dist_cent, _ = nn_cent.kneighbors(u)
        v = dist_cent[:, 1] # pylint: disable=invalid-name

        # computing the response of the hidden units
        phi = pairwise_distances_mahalanobis(X, Y=u, tensor=nn_params.get('metric_tensor', None))
        phi = phi**2
        phi = np.exp(-phi/v**2)

        # applying linear regression to find the best weights
        linreg = LinearRegression()
        linreg.fit(phi, y)
        f = linreg.predict(phi[np.where(y == self.min_label)[0]]) # pylint: disable=invalid-name
        w = linreg.coef_ # pylint: disable=invalid-name

        return u, v, f, w

    def determine_Q(self, X, X_min, nn_params): # pylint: disable=invalid-name
        """
        Calculate the Q value.

        Args:
            X (np.array): all training vectors
            X_min (np.array): all minority samples
            nn_params (dict): the nearest neighbor parameters

        Returns:
            float: the Q value
        """
        # applying nearest neighbors to extract Q values
        n_neighbors = min([self.n_neighbors + 1, len(X)])
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                n_jobs=self.n_jobs,
                                                **nn_params)
        nnmt.fit(X)
        dist, _ = nnmt.kneighbors(X_min)

        Q = np.mean(dist[:, n_neighbors-1])/np.sqrt(len(X[0])) # pylint: disable=invalid-name

        return Q

    def generate_samples(self, X_min, weights, Q, samp_per_iter): # pylint: disable=invalid-name
        """
        Generate samples.

        Args:
            X_min (np.array): the minority samples
            weights (np.array): the density to be used
            Q (float): the Q value
            samp_per_iter (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        base_indices = self.random_state.choice(np.arange(X_min.shape[0]),
                                                samp_per_iter,
                                                p=weights)
        base_vectors = X_min[base_indices]
        lam = (self.random_state.random_sample(base_vectors.shape).T \
                * (2*(1 - weights[base_indices])) - (1 - weights[base_indices])).T

        samples = base_vectors + Q * lam

        return samples

    def determine_Q_and_weights(self, X, y, X_min, nn_params): # pylint: disable=invalid-name
        """
        Determine the Q value and the sampling weights.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            X_min (np.array): all minority samples
            nn_params (dict): the nearest neighbor parameters

        Returns:
            float, np.array: the Q value and the sampling weights
        """
        u, v, f, w = self.calculate_vectors(X, y, X_min, nn_params) # pylint: disable=invalid-name

        Q = self.determine_Q(X, X_min, nn_params) # pylint: disable=invalid-name

        # calculating the sensitivity factors
        I_1 = self.eq_6_vectorized(Q=Q, w=w, u=u, v=v, X=X_min) # pylint: disable=invalid-name
        I_2 = self.eq_8_vectorized(Q=Q, w=w, u=u, v=v, X=X_min) # pylint: disable=invalid-name

        stsm = f**2 - 2*f*I_1 + I_2

        # calculating the sampling weights
        weights = np.abs(stsm)/np.sum(np.abs(stsm))

        return Q, weights

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # number of samples to generate in each iteration
        n_to_sample = self.det_n_to_sample(self.proportion)

        samp_per_iter = max([1, int(n_to_sample/self.n_iter)])

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
            self.metric_tensor_from_nn_params(nn_params, X, y)

        # executing the algorithm
        for _ in range(self.n_iter):
            X_min = X[y == self.min_label]

            Q, weights = self.determine_Q_and_weights(X, y, X_min, nn_params) # pylint: disable=invalid-name

            samples = self.generate_samples(X_min, weights, Q, samp_per_iter)

            X = np.vstack([X, samples])
            y = np.hstack([y, np.repeat(self.min_label, len(samples))])

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'h': self.h,
                'n_iter': self.n_iter,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
