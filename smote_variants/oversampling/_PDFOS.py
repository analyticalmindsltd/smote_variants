import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['PDFOS']

class PDFOS(OverSampling):
    """
    References:
        * BibTex::

            @article{pdfos,
                    title = "PDFOS: PDF estimation based over-sampling for
                                imbalanced two-class problems",
                    journal = "Neurocomputing",
                    volume = "138",
                    pages = "248 - 259",
                    year = "2014",
                    issn = "0925-2312",
                    doi = "https://doi.org/10.1016/j.neucom.2014.02.006",
                    author = "Ming Gao and Xia Hong and Sheng Chen and Chris
                                J. Harris and Emad Khalaf",
                    keywords = "Imbalanced classification, Probability density
                                function based over-sampling, Radial basis
                                function classifier, Orthogonal forward
                                selection, Particle swarm optimisation"
                    }

    Notes:
        * Not prepared for low-rank data.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_density_estimation]

    def __init__(self, 
                 proportion=1.0, 
                 *,
                 n_jobs=1, 
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
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
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def _sample_by_kernel_density_estimation(self,
                                             X,
                                             n_to_sample,
                                             n_optimize=100):
        """
        Sample n_to_sample instances by kernel density estimation

        Args:
            X_min (np.array): minority data
            n_to_sample (int): number of instances to sample
            n_optimize (int): number of vectors used for the optimization
                                process
        """
        # dimensionality of the data
        m = len(X[0])

        # computing the covariance matrix of the data
        S = np.cov(X, rowvar=False)
        message = "Condition number of covariance matrix: %f"
        message = message % np.linalg.cond(S)
        _logger.info(self.__class__.__name__ + ": " + message)

        message = "Inputs size: %d" % len(X)
        _logger.info(self.__class__.__name__ + ": " + message)
        _logger.info(self.__class__.__name__ + ": " + "Input dim: %d" % m)

        S_mrank = np.linalg.matrix_rank(S, tol=1e-2)
        message = "Matrix rank of covariance matrix: %d" % S_mrank
        _logger.info(self.__class__.__name__ + ": " + message)

        # checking the rank of the matrix
        if S_mrank < m:
            message = "The covariance matrix is singular, fixing it by PCA"
            _logger.info(self.__class__.__name__ + ": " + message)
            message = "dim: %d, rank: %d, size: %d" % (m, S_mrank, len(X))
            _logger.info(self.__class__.__name__ + ": " + message)

            n_components = max([min([S_mrank, len(X)])-1, 2])
            if n_components == len(X[0]):
                return X.copy()

            pca = PCA(n_components=n_components)
            X_low_dim = pca.fit_transform(X)
            X_samp = self._sample_by_kernel_density_estimation(
                X_low_dim, n_to_sample, n_optimize)
            return pca.inverse_transform(X_samp)

        S_inv = np.linalg.inv(S)
        det = np.linalg.det(S)

        _logger.info(self.__class__.__name__ + ": " + "Determinant: %f" % det)

        def eq_9(i, j, sigma, X):
            """
            Eq (9) in the paper
            """
            tmp = np.dot(np.dot((X[j] - X[i]), S_inv), (X[j] - X[i]))
            numerator = (np.sqrt(2)*sigma)**(-m)*np.exp(-(1/(4*sigma**2))*tmp)
            denominator = ((2*np.pi)**(m/2))
            return numerator/denominator

        def eq_5(i, j, sigma, X):
            """
            Eq (5) in the paper
            """
            tmp = np.dot(np.dot((X[j] - X[i]), S_inv), (X[j] - X[i]))
            numerator = sigma**(-m)*np.exp(-(1/(2*sigma**2))*tmp)
            denominator = ((2.0*np.pi)**(m/2))
            return numerator/denominator

        def eq_5_0(sigma, X):
            """
            Eq (5) with the same vectors feeded in
            """
            return sigma**(-m)/((2.0*np.pi)**(m/2))

        def eq_8(i, j, sigma, X):
            """
            Eq (8) in the paper
            """
            e9 = eq_9(i, j, sigma, X)
            e5 = eq_5(i, j, sigma, X)
            return e9 - 2*e5

        def M(sigma, X):
            """
            Eq (7) in the paper
            """
            total = 0.0
            for i in range(len(X)):
                for j in range(len(X)):
                    total = total + eq_8(i, j, sigma, X)

            a = total/len(X)**2
            b = 2.0*eq_5_0(sigma, X)/len(X)
            return a + b

        # finding the best sigma parameter
        best_sigma = 0
        error = np.inf
        # the dataset is reduced to make the optimization more efficient
        domain = range(len(X))
        n_to_choose = min([len(X), n_optimize])
        X_reduced = X[self.random_state.choice(domain,
                                               n_to_choose,
                                               replace=False)]

        # we suppose that the data is normalized, thus, this search space
        # should be meaningful
        for sigma in np.logspace(-5, 2, num=20):
            e = M(sigma, X_reduced)
            if e < error:
                error = e
                best_sigma = sigma
        _logger.info(self.__class__.__name__ + ": " +
                     "best sigma found: %f" % best_sigma)

        # generating samples according to the
        samples = []
        for _ in range(n_to_sample):
            idx = self.random_state.randint(len(X))
            samples.append(self.random_state.multivariate_normal(
                X[idx], best_sigma*S))

        return np.vstack(samples)

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

        # scaling the data to aid numerical stability
        ss = StandardScaler()
        X_ss = ss.fit_transform(X)

        X_min = X_ss[y == self.min_label]

        # generating samples by kernel density estimation
        samples = self._sample_by_kernel_density_estimation(X_min,
                                                            n_to_sample,
                                                            n_optimize=100)

        return (np.vstack([X, ss.inverse_transform(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
