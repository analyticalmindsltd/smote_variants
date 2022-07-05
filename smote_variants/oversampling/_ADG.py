import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['ADG']

class ADG(OverSampling):
    """
    References:
        * BibTex::

            @article{adg,
                    author = {Pourhabib, A. and Mallick, Bani K. and Ding, Yu},
                    year = {2015},
                    month = {16},
                    pages = {2695--2724},
                    title = {A Novel Minority Cloning Technique for
                                Cost-Sensitive Learning},
                    volume = {16},
                    journal = {Journal of Machine Learning Research}
                    }

    Notes:
        * This method has a lot of parameters, it becomes fairly hard to
            cross-validate thoroughly.
        * Fails if matrix is singular when computing alpha_star, fixed
            by PCA.
        * Singularity might be caused by repeating samples.
        * Maintaining the kernel matrix becomes unfeasible above a couple
            of thousand vectors.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 *,
                 kernel='inner',
                 lam=1.0,
                 mu=1.0,
                 k=12,
                 gamma=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            kernel (str): 'inner'/'rbf_x', where x is a float, the bandwidth
            lam (float): lambda parameter of the method
            mu (float): mu parameter of the method
            k (int): number of samples to generate in each iteration
            gamma (float): gamma parameter of the method
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)

        if kernel != 'inner' and not kernel.startswith('rbf'):
            raise ValueError(self.__class__.__name__ + ": " +
                             'Kernel function %s not supported' % kernel)
        elif kernel.startswith('rbf'):
            par = float(kernel.split('_')[-1])
            if par <= 0.0:
                raise ValueError(self.__class__.__name__ + ": " +
                                 'Kernel parameter %f is not supported' % par)

        self.check_greater(lam, 'lam', 0)
        self.check_greater(mu, 'mu', 0)
        self.check_greater_or_equal(k, 'k', 1)
        self.check_greater(gamma, 'gamma', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.kernel = kernel
        self.lam = lam
        self.mu = mu
        self.k = k
        self.gamma = gamma
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
                                  'kernel': ['inner', 'rbf_0.5',
                                             'rbf_1.0', 'rbf_2.0'],
                                  'lam': [1.0, 2.0],
                                  'mu': [1.0, 2.0],
                                  'k': [12],
                                  'gamma': [1.0, 2.0]}
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        def bic_score(kmeans, X):
            """
            Compute BIC score for clustering

            Args:
                kmeans (sklearn.KMeans): kmeans object
                X (np.matrix):  clustered data

            Returns:
                float: bic value

            Inspired by https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans
            """  # noqa
            # extract descriptors of the clustering
            cluster_centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_
            n_clusters = kmeans.n_clusters
            n_in_clusters = np.bincount(cluster_labels)
            N, d = X.shape

            # compute variance for all clusters beforehand

            def sum_norm_2(i):
                return np.sum(np.linalg.norm(X[cluster_labels == i] -
                                             cluster_centers[i])**2)

            cluster_variances = [sum_norm_2(i) for i in range(n_clusters)]
            term_0 = (1.0)/((N - n_clusters) * d)
            term_1 = np.sum(cluster_variances)
            clustering_variance = term_0 * term_1

            const_term = 0.5 * n_clusters * np.log(N) * (d+1)

            def bic_comp(i):
                term_0 = n_in_clusters[i] * np.log(n_in_clusters[i])
                term_1 = n_in_clusters[i] * np.log(N)
                term_2 = (((n_in_clusters[i] * d) / 2)
                          * np.log(2*np.pi*clustering_variance))
                term_3 = ((n_in_clusters[i] - 1) * d / 2)

                return term_0 - term_1 - term_2 - term_3

            bic = np.sum([bic_comp(i) for i in range(n_clusters)]) - const_term

            return bic

        def xmeans(X, r=(1, 10)):
            """
            Clustering with BIC based n_cluster selection

            Args:
                X (np.matrix): data to cluster
                r (tuple): lower and upper bound on the number of clusters

            Returns:
                sklearn.KMeans: clustering with lowest BIC score
            """
            best_bic = np.inf
            best_clustering = None

            # do clustering for all n_clusters in the specified range
            for k in range(r[0], min([r[1], len(X)])):
                kmeans = KMeans(n_clusters=k,
                                random_state=self._random_state_init).fit(X)

                bic = bic_score(kmeans, X)
                if bic < best_bic:
                    best_bic = bic
                    best_clustering = kmeans

            return best_clustering

        def xgmeans(X, r=(1, 10)):
            """
            Gaussian mixture with BIC to select the optimal number
            of components

            Args:
                X (np.matrix): data to cluster
                r (tuple): lower and upper bound on the number of components

            Returns:
                sklearn.GaussianMixture: Gaussian mixture model with the
                                            lowest BIC score
            """
            best_bic = np.inf
            best_mixture = None

            # do model fitting for all n_components in the specified range
            for k in range(r[0], min([r[1], len(X)])):
                gmm = GaussianMixture(
                    n_components=k, random_state=self._random_state_init).fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_mixture = gmm

            return best_mixture

        def evaluate_matrices(X, y, kernel=np.inner):
            """
            The function evaluates the matrices specified in the method.

            Args:
                X (np.matrix): features
                y (np.array): target labels
                kernel (function): the kernel function to be used

            Returns:
                np.matrix, np.matrix, int, int, np.matrix, np.array,
                np.matrix, np.matrix, np.matrix
                np.array, np.matrix, np.matrix, np.matrix, np.matrix:
                    X_minux, X_plus, l_minus, l_plus, X, y, K, M_plus, M_minus,
                    M, K_plus, K_minus, N_plus, n_minus using the notations of
                    the paper, X and y are ordered by target labels
            """
            X_minus = X[y == self.maj_label]
            X_plus = X[y == self.min_label]
            l_minus = len(X_minus)
            l_plus = len(X_plus)

            X = np.vstack([X_minus, X_plus])
            y = np.hstack([np.array([self.maj_label]*l_minus),
                           np.array([self.min_label]*l_plus)])

            K = pairwise_distances(X, X, metric=kernel)
            M_plus = np.mean(K[:, len(X_minus):], axis=1)
            M_minus = np.mean(K[:, :len(X_minus)], axis=1)
            M = np.dot(M_minus - M_plus, M_minus - M_plus)

            K_minus = K[:, :len(X_minus)]
            K_plus = K[:, len(X_minus):]

            return (X_minus, X_plus, l_minus, l_plus, X, y, K,
                    M_plus, M_minus, M, K_plus, K_minus)

        # Implementation of the technique, following the steps and notations
        # of the paper
        q = n_to_sample

        # instantiating the proper kernel function, the parameter of the RBF
        # is supposed to be the denominator in the Gaussian
        if self.kernel == 'inner':
            kernel_function = np.inner
        else:
            kf = self.kernel.split('_')
            if kf[0] == 'rbf':
                d = float(kf[1])
                def kernel_function(
                    x, y): return np.exp(-np.linalg.norm(x - y)**2/d)

        # Initial evaluation of the matrices
        (X_minus, X_plus, l_minus, l_plus, X, y, K, M_plus, M_minus,
         M, K_plus, K_minus) = evaluate_matrices(X,
                                                 y,
                                                 kernel=kernel_function)
        # The computing of N matrix is factored into two steps, computing
        # N_plus and N_minus this is used to improve efficiency
        K_plus2 = np.dot(K_plus, K_plus.T)
        K_plus_sum = np.sum(K_plus, axis=1)
        K_plus_diad = np.outer(K_plus_sum, K_plus_sum)/l_plus

        K_minus2 = np.dot(K_minus, K_minus.T)
        K_minus_sum = np.sum(K_minus, axis=1)
        K_minus_diad = np.outer(K_minus_sum, K_minus_sum)/l_minus

        N = K_plus2 - K_plus_diad + K_minus2 - K_minus_diad

        X_plus_hat = X_plus.copy()
        l_minus = len(X_minus)

        early_stop = False
        total_added = 0
        # executing the sample generation
        while q > 1:
            _logger.info(self.__class__.__name__ + ": " +
                         "Starting iteration with q=%d" % q)
            # step 1
            clusters = xmeans(X_plus_hat)
            l_c = np.array([np.sum(clusters.labels_ == i)
                            for i in range(clusters.n_clusters)])

            # step 2
            k_c = ((1.0/l_c)/(np.sum(1.0/l_c))*self.k).astype(int)
            k_c[k_c == 0] = 1
            lam_c, mu_c = self.lam/l_c, self.mu/l_c

            # step 3
            omega = - np.sum([k_c[i]*(lam_c[i])**2/(4*mu_c[i]**2)
                              for i in range(len(k_c))])
            nu_c = - 0.5*k_c*lam_c
            M_plus_c = [np.mean(K[:, np.arange(len(X_minus), len(X))[
                clusters.labels_ == i]]) for i in range(len(k_c))]

            # step 4
            A = (M - self.gamma*N) - omega*K
            b = np.sum([(M_minus - M_plus_c[i])*nu_c[i]
                        for i in range(len(k_c))], axis=0)
            try:
                alpha_star = np.linalg.solve(A, b)
            except Exception as e:
                # handling the issue of singular matrix
                _logger.warning(self.__class__.__name__ +
                                ": " + "Singular matrix")
                # deleting huge data structures
                if q == n_to_sample:
                    if len(X[0]) == 1:
                        return None, None
                    K, K_plus, K_minus = None, None, None
                    n_components = int(np.sqrt(len(X[0])))
                    pca = PCA(n_components=n_components).fit(X)

                    message = "reducing dimensionality to %d" % n_components
                    _logger.warning(self.__class__.__name__ + ": " + message)
                    X_trans = pca.transform(X)
                    adg = ADG(proportion=self.proportion,
                              kernel=self.kernel,
                              lam=self.lam,
                              mu=self.mu,
                              k=self.k,
                              gamma=self.gamma,
                              random_state=self._random_state_init)
                    X_samp, y_samp = adg.sample(X_trans, y)
                    if X_samp is not None:
                        return pca.inverse_transform(X_samp), y_samp
                    else:
                        return X.copy(), y.copy()
                else:
                    q = int(q/2)
                continue

            # step 5
            mixture = xgmeans(X_plus)

            # step 6
            try:
                Z = mixture.sample(q)[0]
            except Exception as e:
                message = "sampling error in sklearn.mixture.GaussianMixture"
                _logger.warning(
                    self.__class__.__name__ + ": " + message)
                return X.copy(), y.copy()

            # step 7
            # computing the kernel matrix of generated samples with all samples
            K_10 = pairwise_distances(Z, X, metric=kernel_function)
            mask_inner_prod = np.where(np.inner(K_10, alpha_star) > 0)[0]
            Z_hat = Z[mask_inner_prod]

            if len(Z_hat) == 0:
                q = int(q/2)
                continue

            _logger.info(self.__class__.__name__ + ": " +
                         "number of vectors added: %d/%d" % (len(Z_hat), q))

            # step 8
            # this step is not used for anything, the identified clusters are
            # only used in step 13 of the paper, however, the values set
            # (M_plus^c) are overwritten in step 3 of the next iteration

            # step 9
            X_plus_hat = np.vstack([X_plus_hat, Z_hat])
            l_plus = len(X_plus_hat)

            # step 11 - 16
            # these steps have been reorganized a bit for efficient
            # calculations

            pairwd = pairwise_distances(Z_hat, Z_hat, metric=kernel_function)
            K = np.block([[K, K_10[mask_inner_prod].T],
                          [K_10[mask_inner_prod], pairwd]])

            K_minus = K[:, :l_minus]
            K_plus = K[:, l_minus:]

            # step 10
            X = np.vstack([X_minus, X_plus_hat])
            y = np.hstack([y, np.repeat(self.min_label, len(Z_hat))])

            if early_stop is True:
                break

            M_plus = np.mean(K_plus, axis=1)
            M_minus = np.mean(K_minus, axis=1)

            # step 13 is already involved in the core of the loop
            M = np.dot(M_minus - M_plus, M_minus - M_plus)

            l_new = len(Z_hat)
            total_added = total_added + l_new

            K_minus2_01 = np.dot(K_minus[:-l_new:], K_minus[-l_new:].T)
            K_minus2 = np.block([[K_minus2, K_minus2_01],
                                 [K_minus2_01.T, np.dot(K_minus[-l_new:],
                                                        K_minus[-l_new:].T)]])
            K_minus_sum = M_minus*len(K_minus)

            K_plus2 = K_plus2 + np.dot(K_plus[:-l_new, l_new:],
                                       K_plus[:-l_new, l_new:].T)

            K_plus2_01 = np.dot(K_plus[:-l_new], K_plus[-l_new:].T)

            K_plus2 = np.block([[K_plus2, K_plus2_01],
                                [K_plus2_01.T, np.dot(K_plus[-l_new:],
                                                      K_plus[-l_new:].T)]])

            K_plus_sum = M_plus*len(K_plus)

            N = K_plus2 - np.outer(K_plus_sum/l_plus, K_plus_sum) + \
                K_minus2 - np.outer(K_minus_sum/l_minus, K_minus_sum)

            # step 17
            if l_new/total_added < 0.01:
                early_stop = True
            else:
                q = int(q/2)

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'kernel': self.kernel,
                'lam': self.lam,
                'mu': self.mu,
                'k': self.k,
                'gamma': self.gamma,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
