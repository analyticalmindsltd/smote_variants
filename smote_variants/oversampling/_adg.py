"""
This module implements the ADG method.
"""

import warnings

import numpy as np
from numpy.linalg import LinAlgError

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances

from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['ADG']

def bic_score(kmeans, X):
    """
    Compute BIC score for clustering

    Args:
        kmeans (sklearn.KMeans): kmeans object
        X (np.array):  clustered data

    Returns:
        float: bic value

    Inspired by https://stats.stackexchange.com/questions/90769/
                        using-bic-to-estimate-the-number-of-k-in-kmeans
    """

    # extract descriptors of the clustering
    cluster_centers = kmeans.cluster_centers_
    #n_clusters = kmeans.n_clusters
    n_clusters = np.unique(kmeans.labels_).shape[0]
    n_in_clusters = np.bincount(kmeans.labels_)
    n_samples, n_dim = X.shape

    # compute variance for all clusters beforehand

    def sum_norm_2(idx):
        return np.sum(np.linalg.norm(X[kmeans.labels_ == idx] -
                                        cluster_centers[idx])**2)

    cluster_variances = [sum_norm_2(i) for i in range(n_clusters)]
    term_0 = (1.0)/((n_samples - n_clusters) * n_dim)
    term_1 = np.sum(cluster_variances)
    clustering_variance = term_0 * term_1

    const_term = 0.5 * n_clusters * np.log(n_samples) * (n_dim+1)

    def bic_comp(idx):
        term_0 = n_in_clusters[idx] * np.log(n_in_clusters[idx])
        term_1 = n_in_clusters[idx] * np.log(n_samples)
        if clustering_variance == 0:
            return np.inf
        term_2 = (((n_in_clusters[idx] * n_dim) / 2)
                    * np.log(2*np.pi*clustering_variance))
        term_3 = ((n_in_clusters[idx] - 1) * n_dim / 2)

        return term_0 - term_1 - term_2 - term_3

    bic = np.sum([bic_comp(idx) for idx in range(n_clusters)
                                if n_in_clusters[idx] > 0]) - const_term

    return bic

def xmeans(X, rng=(1, 10), random_state=None):
    """
    Clustering with BIC based n_cluster selection

    Args:
        X (np.array): data to cluster
        rng (tuple): lower and upper bound on the number of clusters
        random_state (int/None): random state to use

    Returns:
        sklearn.KMeans: clustering with lowest BIC score
    """
    best_bic = np.inf
    best_clustering = None

    # do clustering for all n_clusters in the specified range
    for n_clusters in range(rng[0], min([rng[1], len(X)])):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kmeans = KMeans(n_clusters=n_clusters,
                            random_state=random_state).fit(X)

        bic = bic_score(kmeans, X)
        if bic < best_bic:
            best_bic = bic
            best_clustering = kmeans

    return best_clustering

def xgmeans(X, rng=(1, 10), random_state=None):
    """
    Gaussian mixture with BIC to select the optimal number
    of components

    Args:
        X (np.array): data to cluster
        rng (tuple): lower and upper bound on the number of components
        random_state (int/None): random state to use

    Returns:
        sklearn.GaussianMixture: Gaussian mixture model with the
                                    lowest BIC score
    """
    best_bic = np.inf
    best_mixture = None

    # do model fitting for all n_components in the specified range
    for n_components in range(rng[0], min([rng[1], len(X)])):
        gmm = GaussianMixture(n_components=n_components,
                                random_state=random_state).fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_mixture = gmm

    return best_mixture

def evaluate_matrices(X, y, kernel=np.inner, maj_label=0, min_label=1):
    """
    The function evaluates the partial_results specified in the method.

    Args:
        X (np.array): features
        y (np.array): target labels
        kernel (function): the kernel function to be used

    Returns:
        np.array, np.array, int, int, np.array, np.array,
        np.array, np.array, np.array
        np.array, np.array, np.array, np.array, np.array:
            X_minux, X_plus, l_minus, l_plus, X, y, K, M_plus, M_minus,
            M, K_plus, K_minus, N_plus, n_minus using the notations of
            the paper, X and y are ordered by target labels
    """
    x_minus = X[y == maj_label]
    x_plus = X[y == min_label]

    X = np.vstack([x_minus, x_plus])
    y = np.hstack([np.array([maj_label]*len(x_minus)),
                    np.array([min_label]*len(x_plus))])

    k = pairwise_distances(X, X, metric=kernel) # pylint: disable=invalid-name
    m_plus = np.mean(k[:, len(x_minus):], axis=1)
    m_minus = np.mean(k[:, :len(x_minus)], axis=1)
    m = np.dot(m_minus - m_plus, m_minus - m_plus) # pylint: disable=invalid-name

    k_minus = k[:, :len(x_minus)]
    k_plus = k[:, len(x_minus):]

    partial_results = {'X_minus': x_minus,
                'X_plus': x_plus,
                'K': k,
                'M_plus': m_plus,
                'M_minus': m_minus,
                'M': m,
                'K_plus': k_plus,
                'K_minus': k_minus,
                'l_minus': len(x_minus),
                'l_plus': len(x_plus)}

    return (partial_results, X, y)


def generate_kernel_function(specification):
    """
    Generate the kernel function.

    Args:
        specification (str): the kernel specification

    Returns:
        callable: the kernel function
    """
    if specification == 'inner':
        kernel_function = np.inner
    else:
        kernel_function = None
        specs = specification.split('_')
        if specs[0] == 'rbf':
            bandwidth = float(specs[1])
            def kernel_function(x_vector, y_vector):
                norm = np.linalg.norm(x_vector - y_vector)
                return np.exp(-norm**2/bandwidth)

    return kernel_function

class ADGSingularMatrixException(Exception):
    """
    Exception indicating a singular matrix problem.
    """

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
                 random_state=None,
                 **_kwargs):
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
        super().__init__(random_state=random_state, checks=None)

        self.check_greater_or_equal(proportion, "proportion", 0)

        if kernel != 'inner' and not kernel.startswith('rbf'):
            raise ValueError(f"{self.__class__.__name__}: Kernel function "\
                                f"{kernel} not supported")
        if kernel.startswith('rbf'):
            par = float(kernel.split('_')[-1])
            if par <= 0.0:
                raise ValueError(f"{self.__class__.__name__}: Kernel parameter "\
                                    f"{par} is not supported")

        self.check_greater(lam, 'lam', 0)
        self.check_greater(mu, 'mu', 0)
        self.check_greater_or_equal(k, 'k', 1)
        self.check_greater(gamma, 'gamma', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.kernel = kernel
        self.lam = lam
        self.mu = mu # pylint: disable=invalid-name
        self.k = k # pylint: disable=invalid-name
        self.gamma = gamma
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
                                  'kernel': ['inner', 'rbf_0.5',
                                             'rbf_1.0', 'rbf_2.0'],
                                  'lam': [1.0, 2.0],
                                  'mu': [1.0, 2.0],
                                  'k': [12],
                                  'gamma': [1.0, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def initialize_matrices(self, *, partial_results):
        """
        Initializes the base partial_results.

        Args:
            np.arrays by names

        Returns:
            np.arrays: the initialized partial_results
        """
        l_plus = partial_results['l_plus']
        k_plus = partial_results['K_plus']
        k_minus = partial_results['K_minus']
        x_plus = partial_results['X_plus']
        x_minus = partial_results['X_minus']

        k_plus2 = np.dot(k_plus, k_plus.T)
        k_plus_sum = np.sum(k_plus, axis=1)
        k_plus_diad = np.outer(k_plus_sum, k_plus_sum)/l_plus

        k_minus2 = np.dot(k_minus, k_minus.T)
        k_minus_sum = np.sum(k_minus, axis=1)
        k_minus_diad = np.outer(k_minus_sum,
                                k_minus_sum)/partial_results['l_minus']

        n = k_plus2 - k_plus_diad + k_minus2 - k_minus_diad # pylint: disable=invalid-name

        x_plus_hat = x_plus.copy()

        partial_results['K_plus2'] = k_plus2
        partial_results['K_plus_sum'] = k_plus_sum
        partial_results['K_minus2'] = k_minus2
        partial_results['K_minus_sum'] = k_minus_sum
        partial_results['N'] = n
        partial_results['X_plus_hat'] = x_plus_hat
        partial_results['l_minus'] = len(x_minus)

        return partial_results

    def sample_with_n_components(self, X, y, n_components):
        """
        Rerun the sampling in a lower dimensional space.

        Args:
            X (np.array): features
            y (np.array): targets
            n_components (int): dimensionality of the feature space

        Returns:
            np.array, np.array: the resampled X and y
        """
        _logger.warning("%s: reducing dimensionality to %d",
                        self.__class__.__name__, n_components)
        #if n_components == 0:
        #    return None, None

        pca = PCA(n_components=n_components).fit(X)

        x_trans = pca.transform(X)

        adg = ADG(proportion=self.proportion,
                    kernel=self.kernel,
                    lam=self.lam,
                    mu=self.mu,
                    k=self.k,
                    gamma=self.gamma,
                    random_state=self._random_state_init)

        X_samp, y_samp = adg.sample(x_trans, y)

        return pca.inverse_transform(X_samp), y_samp

    def update_partial_results(self, *, partial_results):
        """
        Update partial_results.

        Args:
            np.arrays: the constants and partial_results used in the paper

        Returns:
            np.array: the updated partial_results
        """
        # M_plus is not updated as it is not used

        k_plus = partial_results['K_plus']
        k_minus = partial_results['K_minus']
        k_minus2 = partial_results['K_minus2']
        k_plus2 = partial_results['K_plus2']
        l_new = partial_results['l_new']

        m_plus = np.mean(k_plus, axis=1)
        m_minus = np.mean(k_minus, axis=1)

        # step 13 is already involved in the core of the loop
        m = np.dot(m_minus - m_plus, m_minus - m_plus) # pylint: disable=invalid-name

        corner = np.dot(k_minus[:-l_new:], k_minus[-l_new:].T)
        k_minus2 = np.block([[k_minus2, corner],
                                [corner.T, np.dot(k_minus[-l_new:],
                                                    k_minus[-l_new:].T)]])
        k_minus_sum = m_minus*len(k_minus)

        k_plus2 = k_plus2 + np.dot(k_plus[:-l_new, l_new:],
                                    k_plus[:-l_new, l_new:].T)

        corner = np.dot(k_plus[:-l_new], k_plus[-l_new:].T)

        k_plus2 = np.block([[k_plus2, corner],
                            [corner.T, np.dot(k_plus[-l_new:],
                                                    k_plus[-l_new:].T)]])

        k_plus_sum = m_plus*len(k_plus)

        n = k_plus2 - np.outer(k_plus_sum/partial_results['l_plus'], # pylint: disable=invalid-name
                                k_plus_sum) + \
            k_minus2 - np.outer(k_minus_sum/partial_results['l_minus'],
                                k_minus_sum)

        partial_results['M_plus'] = m_plus
        partial_results['M_minus'] = m_minus
        partial_results['M'] = m
        partial_results['K_minus2'] = k_minus2
        partial_results['K_minus_sum'] = k_minus_sum
        partial_results['K_plus2'] = k_plus2
        partial_results['K_plus_sum'] = k_plus_sum
        partial_results['N'] = n

        return partial_results


    def determine_a(self, *, partial_results, k_c, lam_c, mu_c):
        """
        Determine the matrix A

        Args:
            the quantities required for A

        Returns:
            np.array: the matrix A
        """
        m = partial_results['M'] # pylint: disable=invalid-name
        n = partial_results['N'] # pylint: disable=invalid-name
        k = partial_results['K'] # pylint: disable=invalid-name

        # step 3, 4
        omega = - np.sum([k_c[i]*(lam_c[i])**2/(4*mu_c[i]**2)
                            for i in range(len(k_c))])
        A = (m - self.gamma*n) - omega*k # pylint: disable=invalid-name

        return A

    def determine_b(self, *, partial_results, k_c, lam_c, X, clusters):
        """
        Determine the vector b

        Args:
            the quantities required for vector b

        Returns:
            np.array: the vector b
        """
        x_minus = partial_results['X_minus']
        k = partial_results['K'] # pylint: disable=invalid-name
        m_minus = partial_results['M_minus']

        # step 3, 4
        nu_c = - 0.5*k_c*lam_c
        m_plus_c = [np.mean(k[:, np.arange(len(x_minus), len(X))[
            clusters.labels_ == i]]) for i in range(len(k_c))]

        b = np.sum([(m_minus - m_plus_c[i])*nu_c[i] # pylint: disable=invalid-name
                    for i in range(len(k_c))], axis=0)

        return b

    def step_1_2_3_4(self, *, partial_results, X):
        """
        Implements steps 1, 2, 3 and 4 of the paper.

        Args:
            np.arrays: the constants and partial_results used in the paper

        Returns:
            np.array: the updated partial_results
        """
        x_plus_hat = partial_results['X_plus_hat']

        # step 1
        clusters = xmeans(x_plus_hat, random_state = self._random_state_init)
        l_c = np.array([np.sum(clusters.labels_ == i)
                        for i in range(clusters.n_clusters)])

        # step 2
        k_c = ((1.0/l_c)/(np.sum(1.0/l_c))*self.k).astype(int)
        k_c[k_c == 0] = 1
        lam_c, mu_c = self.lam/l_c, self.mu/l_c

        return (self.determine_a(partial_results=partial_results, k_c=k_c, lam_c=lam_c,
                                    mu_c=mu_c),
                self.determine_b(partial_results=partial_results, k_c=k_c, lam_c=lam_c,
                                    X=X, clusters=clusters))

    def step_5_6_7(self, *, partial_results, q, X, kernel_function, alpha_star):
        """
        Implements steps 5, 6, 7 of the paper.

        Args:
            np.arrays: the constants and partial_results used in the paper

        Returns:
            np.array: the updated partial_results
        """
        x_plus = partial_results['X_plus']

        # step 5
        mixture = xgmeans(x_plus, random_state=self._random_state_init)

        # step 6
        z = mixture.sample(q)[0] # pylint: disable=invalid-name

        # step 7
        # computing the kernel matrix of generated samples with all samples
        k_10 = pairwise_distances(z, X, metric=kernel_function)
        mask_inner_prod = np.where(np.inner(k_10, alpha_star) > 0)[0]
        z_hat = z[mask_inner_prod]

        partial_results['Z_hat'] = z_hat
        partial_results['mask_inner_prod'] = mask_inner_prod
        partial_results['K_10'] = k_10

    def step_9_16(self, *, partial_results, kernel_function, y):
        """
        Implements steps 9-16 of the paper.

        Args:
            np.arrays: the constants and partial_results used in the paper

        Returns:
            np.array: the updated partial_results
        """
        # step 8
        # this step is not used for anything, the identified clusters are
        # only used in step 13 of the paper, however, the values set
        # (M_plus^c) are overwritten in step 3 of the next iteration

        if partial_results['Z_hat'].shape[0] == 0:
            raise ValueError("Z_hat is empty")

        x_minus = partial_results['X_minus']
        x_plus_hat = partial_results['X_plus_hat']
        k = partial_results['K'] # pylint: disable=invalid-name
        k_10 = partial_results['K_10']
        l_minus = partial_results['l_minus']
        z_hat = partial_results['Z_hat']
        mask_inner_prod = partial_results['mask_inner_prod']

        # step 9
        x_plus_hat = np.vstack([x_plus_hat, z_hat])

        # step 11 - 16
        # these steps have been reorganized a bit for efficient
        # calculations

        pairwd = pairwise_distances(z_hat, z_hat, metric=kernel_function)
        k = np.block([[k, k_10[mask_inner_prod].T], # pylint: disable=invalid-name
                        [k_10[mask_inner_prod], pairwd]])

        k_minus = k[:, :l_minus]
        k_plus = k[:, l_minus:]

        # step 10
        X = np.vstack([x_minus, x_plus_hat])
        y = np.hstack([y, np.repeat(self.min_label, len(z_hat))])

        partial_results['X_plus_hat'] = x_plus_hat
        partial_results['K'] = k
        partial_results['K_minus'] = k_minus
        partial_results['K_plus'] = k_plus
        partial_results['l_plus'] = len(x_plus_hat)
        partial_results['l_new'] = len(partial_results['Z_hat'])

        return X, y

    def solve(self,
                    A, # pylint: disable=invalid-name
                    b  # pylint: disable=invalid-name
                    ):
        """
        Solve a linear equation and throw exception if doesnt work.

        Args:
            A (np.array): the coefficients
            b (np.array): the right hand side

        Returns:
            np.array: the solution
        """
        try:
            return np.linalg.solve(A, b)
        except LinAlgError as exc:
            raise ADGSingularMatrixException() from exc

    def check_early_stopping(self, fraction):
        """
        Checking the early stopping condition.
        """
        if fraction < 0.01:
            raise ValueError("Early stopping.")

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # Implementation of the technique, following the steps and notations
        # of the paper
        n_to_sample = self.det_n_to_sample(self.proportion)

        q = n_to_sample # pylint: disable=invalid-name

        if q == 0:
            return self.return_copies(X, y, "no need for sampling")

        # instantiating the proper kernel function, the parameter of the RBF
        # is supposed to be the denominator in the Gaussian
        kernel_function = generate_kernel_function(self.kernel)

        # Initial evaluation of the partial_results
        # M_plus is not updated as it is not used
        # the partial_results dict acts like a dataclass to reduce the clutter
        # of passing around dozens of arrays
        partial_results, X, y = evaluate_matrices(X, y, kernel=kernel_function)

        # The computing of N matrix is factored into two steps, computing
        # N_plus and N_minus this is used to improve efficiency
        partial_results = self.initialize_matrices(partial_results=partial_results)

        total_added = 0

        alpha_star = None

        q = q * 2 # pylint: disable=invalid-name

        try:
            # executing the sample generation
            while q > 1:
                q = int(q/2) # pylint: disable=invalid-name

                _logger.info("%s: Starting iteration with q=%d",
                                self.__class__.__name__ , q)

                A, b = self.step_1_2_3_4(partial_results=partial_results, X=X) # pylint: disable=invalid-name

                alpha_star = self.solve(A, b)

                self.step_5_6_7(partial_results=partial_results,
                                q=q,
                                X=X,
                                kernel_function=kernel_function,
                                alpha_star=alpha_star)

                _logger.info("%s: number of vectors added: %d/%d",
                            self.__class__.__name__, len(partial_results['Z_hat']), q)

                X, y = self.step_9_16(partial_results=partial_results,
                                        kernel_function=kernel_function,
                                        y=y)

                total_added = total_added + len(partial_results['Z_hat'])

                self.check_early_stopping(len(partial_results['Z_hat'])/total_added)

                partial_results = self.update_partial_results(partial_results=partial_results)

        except ADGSingularMatrixException:
            n_components = int(X.shape[1]/2)
            if n_components > 0:
                return self.sample_with_n_components(X, y, n_components)

            return self.return_copies(X, y, "cannot reduce dimensionality any more")
        except ValueError as exc:
            _logger.info("%s: stopping the iteration because of %s",
                        self.__class__.__name__, str(exc))
            return X.copy(), y.copy()

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
                **OverSampling.get_params(self)}
