"""
This module implements the PDFOS method.
"""

import numpy as np
from numpy.linalg import LinAlgError

from scipy.special import expit

from sklearn.preprocessing import StandardScaler

from ..base import RandomStateMixin, coalesce
from ..base import MetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['PDFOS', 'FullRankTransformer', 'PDFOSKDE']

class FullRankTransformer:
    """
    Transforms data into a lower dimensional full rank covariance
    representation
    """
    def __init__(self, eps=0.002):
        """
        Constructor of the object.

        Args:
            eps (float): the tolerance
        """
        self.eps = eps
        self.mean = None
        self.transformation = None
        self.n_dim = None

    def fit(self, X):
        """
        Fits the transformer to the data

        Args:
            X (np.array): all vectors to fit on

        Returns:
            self: the fitted object
        """
        cov = np.cov(X, rowvar=False)
        eigw, eigv = np.linalg.eig(cov)

        eigw = np.real(eigw)
        eigv = np.real(eigv)

        eigv_f = eigv[:, eigw > self.eps]

        self.mean = np.mean(X, axis=0)
        self.transformation = eigv_f
        self.n_dim = eigv_f.shape[1]

        return self

    def transform(self, X):
        """
        Transforms the data

        Args:
            X (np.array): the data to transform

        Returns:
            np.array: the transformed data
        """
        if X is None:
            return None
        return np.dot(X - self.mean, self.transformation)

    def inverse_transform(self, X):
        """
        Inverse transformation

        Args:
            X (np.array): the data to inverse transform

        Returns:
            np.array: the inverse transformed data
        """
        return np.dot(X, self.transformation.T) + self.mean


class PDFOSKDE(RandomStateMixin):
    """
    The KDE in PDFOS
    """
    def __init__(self, metric_learning=None, random_state=None):
        """
        Constructor of the PDFOS KDE estimation

        Args:
            metric_learning (str/None): metric learning method
            random_state (int/None/np.random.RandomState): random state
                                                            initializer
        """
        RandomStateMixin.__init__(self, random_state)
        self.transformer = FullRankTransformer(eps=0.02)
        self.metric_learning = coalesce(metric_learning, 'cov_min')
        self.S = None # pylint: disable=invalid-name
        self.X_base = None

    def eq_5_0(self,
                sigma,
                m # pylint: disable=invalid-name
                ):
        """
        Eq (5) in the paper

        Args:
            sigma (float): the sigma value
            m (int): the dimensionality

        Returns:
            float: the value of the equation
        """
        return sigma**(-m)/((2.0*np.pi)**(m/2))

    def eq_8_vectorized(self,
                        X,
                        sigma,
                        S_inv # pylint: disable=invalid-name
                        ):
        """
        Eq (8) in the paper

        Args:
            X (np.array): all training vectors
            sigma (float): the sigma value
            S_inv (np.array): the inverse covariance matrix

        Returns:
            np.array: the result of the equation for all pairs of vectors
        """
        m = X.shape[1] # pylint: disable=invalid-name

        tmp= (X[:,None] - X)
        tmp = np.einsum('ijk,kl,lji -> ij', tmp, S_inv, tmp.T)

        numerator_9 = (np.sqrt(2)*sigma)**(-m)*expit(-(1/(4*sigma**2))*tmp)
        numerator_5 = sigma**(-m)*expit(-(1/(2*sigma**2))*tmp)
        denominator = ((2*np.pi)**(m/2))
        eq_9 = numerator_9 / denominator
        eq_5 = numerator_5 / denominator

        return eq_9 - 2 * eq_5

    def M(self, # pylint: disable=invalid-name
            sigma,
            X,
            S_inv # pylint: disable=invalid-name
            ):
        """
        Eq (7) in the paper

        Args:
            X (np.array): all training vectors
            sigma (float): the sigma value
            S_inv (np.array): the inverse covariance matrix

        Returns:
            float: the value of the equation
        """
        m = X.shape[1] # pylint: disable=invalid-name
        total = np.sum(self.eq_8_vectorized(X, sigma, S_inv))

        term_a = total/len(X)**2
        term_b = 2.0 * self.eq_5_0(sigma, m)/len(X)
        return term_a + term_b

    def find_best_sigma(self,
                        X,
                        n_optimize,
                        S_inv # pylint: disable=invalid-name
                        ):
        """
        Find the best sigma parameter.

        Args:
            X (np.array): all training vectors
            n_optimize (int): the number of vectors used for
                                the estimation
            S_inv (np.array): the inverse covariance matrix

        Returns:
            float: the best sigma value
        """
        # finding the best sigma parameter
        best_sigma = 0
        error = np.inf
        # the dataset is reduced to make the optimization more efficient
        domain = range(len(X))
        n_to_choose = min([len(X), n_optimize])
        X_reduced = X[self.random_state.choice(domain, # pylint: disable=invalid-name
                                               n_to_choose,
                                               replace=False)]

        # we suppose that the data is normalized, thus, this search space
        # should be meaningful
        for sigma in np.logspace(-5, 2, num=20):
            err = self.M(sigma, X_reduced, S_inv)
            if err < error:
                error = err
                best_sigma = sigma
        #_logger.info("%s: best sigma found %f",
        #                self.__class__.__name__, best_sigma)

        return best_sigma

    def fit(self,
                X,
                X_ml, # pylint: disable=invalid-name
                y_ml,
                n_optimize=100):
        """
        Fitting the kernel density estimator

        Args:
            X (np.array): the training vectors
            X_ml (np.array): potential training vectors for
                                    metric learning
            y_ml (np.array): potential target labels for metric
                                    learning
            n_optimize (int): the number of random samples used
                                for optimization

        Returns:
            obj: the fitted object
        """
        self.transformer.fit(X)
        X_trans = self.transformer.transform(X) # pylint: disable=invalid-name

        X_ml = self.transformer.transform(X_ml)

        metrict = MetricTensor(metric_learning_method=self.metric_learning)
        S_inv = metrict.tensor(X_ml, y_ml) # pylint: disable=invalid-name
        S = np.linalg.inv(S_inv) # pylint: disable=invalid-name

        best_sigma = self.find_best_sigma(X_trans, n_optimize, S_inv)

        self.S = best_sigma * S
        self.X_base = X_trans

        return self

    def sample(self, n_to_sample):
        """
        Draw a random sample from the fitted KDE

        Args:
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        samples_raw = self.random_state.multivariate_normal(np.repeat(0.0, self.X_base.shape[1]),
                                                            self.S,
                                                            size=n_to_sample)

        base_indices = self.random_state.choice(self.X_base.shape[0], n_to_sample)
        unique_indices, unique_counts = np.unique(base_indices, return_counts=True)
        samples = []
        loc = 0
        for idx, base_idx in enumerate(unique_indices):
            samples.append(self.X_base[base_idx] \
                            + samples_raw[loc:(loc + unique_counts[idx])])
            loc = loc + unique_counts[idx]
        samples = np.vstack(samples)
        return self.transformer.inverse_transform(samples)

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
                 metric_learning=None,
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
            metric_learning (str/None): optional metric learning method 'ITML',
                                        'LSML'
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.metric_learning = metric_learning
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

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            return self.return_copies(X, y, n_to_sample)

        # scaling the data to aid numerical stability
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) # pylint: disable=invalid-name

        X_min = X_scaled[y == self.min_label]

        # generating samples by kernel density estimation
        try:
            kde = PDFOSKDE(metric_learning=self.metric_learning,
                            random_state=self._random_state_init).fit(X_min,
                                                                        X_ml=X_scaled,
                                                                        y_ml=y)
        except LinAlgError as exc:
            return self.return_copies(X, y, f"kde fitting did not succeed {exc}")
        samples = kde.sample(n_to_sample)

        return (np.vstack([X, scaler.inverse_transform(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
