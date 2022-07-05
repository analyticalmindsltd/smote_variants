import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances

import scipy.special as sspecial

from .._metric_tensor import (NearestNeighborsWithMetricTensor, 
                                MetricTensor, pairwise_distances_mahalanobis)
from ._OverSampling import OverSampling
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
                 h=10,
                 n_iter=5,
                 n_jobs=1,
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(h, "h", 1)
        self.check_greater_or_equal(n_iter, "n_iter", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.h = h
        self.n_iter = n_iter
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
                                  'n_neighbors': [3, 5],
                                  'h': [2, 5, 10, 20],
                                  'n_iter': [5]}
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

        # number of samples to generate in each iteration
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        samp_per_iter = max([1, int(n_to_sample/self.n_iter)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # executing the algorithm
        for _ in range(self.n_iter):
            X_min = X[y == self.min_label]

            # applying kmeans clustering to find the hidden neurons
            h = min([self.h, len(X_min)])
            kmeans = KMeans(n_clusters=h,
                            random_state=self._random_state_init)
            kmeans.fit(X)

            # extracting the hidden center elements
            u = kmeans.cluster_centers_

            # extracting scale parameters as the distances of closest centers
            nn_cent = NearestNeighborsWithMetricTensor(n_neighbors=2, 
                                                        n_jobs=self.n_jobs, 
                                                        **nn_params)
            nn_cent.fit(u)
            dist_cent, ind_cent = nn_cent.kneighbors(u)
            v = dist_cent[:, 1]

            # computing the response of the hidden units
            phi = pairwise_distances_mahalanobis(u, X, nn_params.get('metric_tensor', None))
            phi = phi**2
            phi = np.exp(-phi/v**2)

            # applying linear regression to find the best weights
            lr = LinearRegression()
            lr.fit(phi, y)
            f = lr.predict(phi[np.where(y == self.min_label)[0]])
            w = lr.coef_

            def eq_6(Q, w, u, v, x):
                """
                Equation 6 in the paper
                """
                tmp_sum = np.zeros(h)
                for i in range(h):
                    a = (x - u[i] + Q)/np.sqrt(2*v[i])
                    b = (x - u[i] - Q)/np.sqrt(2*v[i])
                    tmp_prod = (sspecial.erf(a) - sspecial.erf(b))
                    tmp_sum[i] = np.sqrt(np.pi/2)*v[i]*np.prod(tmp_prod)
                return np.dot(tmp_sum, w)/(2*Q)**len(x)

            def eq_8(Q, w, u, v, x):
                """
                Equation 8 in the paper
                """
                res = 0.0
                for i in range(h):
                    vi2 = v[i]**2
                    for r in range(h):
                        vr2 = v[r]**2
                        a1 = (np.sqrt(2*vi2*vr2*(vi2 + vr2)))

                        a00_v = (vi2 + vr2)*(x + Q)
                        a01_v = vi2*u[r] + vr2*u[i]
                        a0_v = a00_v - a01_v
                        a_v = a0_v/a1

                        b_v = ((vi2 + vr2)*(x - Q) - (vi2*u[r] + vr2*u[i]))/a1
                        tmp_prod = sspecial.erf(a_v) - sspecial.erf(b_v)

                        tmp_a = (np.sqrt(2*vi2*vr2*(vi2 + vr2)) /
                                 (vi2 + vr2))**len(x)
                        norm = np.linalg.norm(u[r] - u[i])
                        tmp_b = np.exp(-0.5 * norm**2/(vi2 + vr2))
                        
                        res = res + tmp_a*tmp_b*np.prod(tmp_prod)*w[i]*w[r]

                return (np.sqrt(np.pi)/(4*Q))**len(x)*res

            # applying nearest neighbors to extract Q values
            n_neighbors = min([self.n_neighbors + 1, len(X)])
            nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
            nn.fit(X)
            dist, ind = nn.kneighbors(X_min)

            Q = np.mean(dist[:, n_neighbors-1])/np.sqrt(len(X[0]))

            # calculating the sensitivity factors
            I_1 = np.array([eq_6(Q, w, u, v, x) for x in X_min])
            I_2 = np.array([eq_8(Q, w, u, v, x) for x in X_min])

            stsm = f**2 - 2*f*I_1 + I_2

            # calculating the sampling weights
            weights = np.abs(stsm)/np.sum(np.abs(stsm))

            n_neighbors = min([len(X_min), self.n_neighbors+1])
            nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                    n_jobs=self.n_jobs, 
                                                    **nn_params)
            nn.fit(X_min)
            dist, ind = nn.kneighbors(X_min)

            samples = []
            for _ in range(samp_per_iter):
                idx = self.random_state.choice(
                    np.arange(len(X_min)), p=weights)
                X_new = X_min[idx].copy()
                for s in range(len(X_new)):
                    lam = self.random_state.random_sample(
                    )*(2*(1 - weights[idx])) - (1 - weights[idx])
                    X_new[s] = X_new[s] + Q*lam
                samples.append(X_new)

            samples = np.vstack(samples)
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
                'random_state': self._random_state_init}
