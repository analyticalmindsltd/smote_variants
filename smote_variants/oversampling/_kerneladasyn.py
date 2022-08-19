"""
This module implements the KernelADASYN method.
"""

import numpy as np

from sklearn.decomposition import PCA

from ..base import fix_density, cov
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['KernelADASYN']

class KernelADASYN(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{kernel_adasyn,
                            author={Tang, B. and He, H.},
                            booktitle={2015 IEEE Congress on Evolutionary
                                        Computation (CEC)},
                            title={KernelADASYN: Kernel based adaptive
                                    synthetic data generation for
                                    imbalanced learning},
                            year={2015},
                            volume={},
                            number={},
                            pages={664-671},
                            keywords={learning (artificial intelligence);
                                        pattern classification;
                                        sampling methods;KernelADASYN;
                                        kernel based adaptive synthetic
                                        data generation;imbalanced
                                        learning;standard classification
                                        algorithms;data distribution;
                                        minority class decision rule;
                                        expensive minority class data
                                        misclassification;kernel based
                                        adaptive synthetic over-sampling
                                        approach;imbalanced data
                                        classification problems;kernel
                                        density estimation methods;Kernel;
                                        Estimation;Accuracy;Measurement;
                                        Standards;Training data;Sampling
                                        methods;Imbalanced learning;
                                        adaptive over-sampling;kernel
                                        density estimation;pattern
                                        recognition;medical and
                                        healthcare data learning},
                            doi={10.1109/CEC.2015.7256954},
                            ISSN={1089-778X},
                            month={May}}

    Notes:
        * The method of sampling was not specified, Markov Chain Monte Carlo
            has been implemented.
        * Not prepared for improperly conditioned covariance matrix.
    """

    categories = [OverSampling.cat_density_estimation,
                  OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 k=5,
                 *,
                 nn_params={},
                 h=1.0,
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
            k (int): number of neighbors in the nearest neighbors component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            h (float): kernel bandwidth
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(k, 'k', 1)
        self.check_greater(h, 'h', 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.k = k # pylint: disable=invalid-name
        self.nn_params = nn_params
        self.h = h # pylint: disable=invalid-name
        self.n_jobs = n_jobs

        self.mcmc_params = {'burn_in' : 1000,
                            'periods' : 50}

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'k': [5, 7, 9],
                                  'h': [0.01, 0.02, 0.05, 0.1, 0.2,
                                        0.5, 1.0, 2.0, 10.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def p_x(self, x_vec, X_min, r_score):
        """
        Returns minority density value at x

        Args:
            x_vec (np.array): feature vector

        Returns:
            float: density value
        """
        result = 1.0/(len(X_min)*self.h)
        result = result*(1.0/(np.sqrt(2*np.pi)*self.h)**len(X_min[0]))

        exp_term = np.exp(-0.5*np.linalg.norm(x_vec - X_min, axis=1)**2/self.h)
        return result*np.inner(r_score, exp_term)

    def mcmc(self, X_min, r_score, covariance, n_to_sample):
        """
        Do the Markov Chain Monte Carlo sampling.

        Args:
            X_min (np.array): the minority samples
            r_score (np.array): the majority scores
            covariance (np.array): the covariance matrix
            n_to_sample (int): the number of samples to generate
        """
        samples = []
        iteration = 0

        # parameters of the Monte Carlo sampling


        # starting Markov-Chain Monte Carlo for sampling
        x_old = X_min[self.random_state.choice(np.where(r_score > 0)[0])]
        p_old = self.p_x(x_old, X_min, r_score)

        # Cholesky decomposition
        L_mat = np.linalg.cholesky(covariance) # pylint: disable=invalid-name

        while len(samples) < n_to_sample:
            x_new = x_old + \
                np.dot(self.random_state.normal(size=len(x_old)), L_mat)
            p_new = self.p_x(x_new, X_min, r_score)

            alpha = p_new/p_old
            if self.random_state.random_sample() < alpha:
                x_old = x_new
                p_old = p_new

            iteration = iteration + 1
            if (iteration % self.mcmc_params['periods'] == 0
                                and iteration > self.mcmc_params['burn_in']):
                samples.append(x_old)

        return np.vstack(samples)

    def iterated_sampling(self, *,
                        covariance,
                        X, y,
                        X_min,
                        r_score,
                        n_to_sample):
        """
        Try to carry out the sampling and reduce the dimensions if it
        does not succeed.

        Args:
            covariance (np.array): the covariance matrix
            X (np.array): the feature vectors
            y (np.array): the target labels
            X_min (np.array): the minority samples
            r_score (np.array): the majority score
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array, np.array: the oversampled dataset
        """
        if len(covariance) > 1 and np.linalg.cond(covariance) > 10000:
            _logger.info("%s: reducing dimensions due to improperly "\
                            "conditioned covariance matrix",
                        self.__class__.__name__)

            if X.shape[1] <= 2:
                return self.return_copies(X, y, "matrix is ill conditioned")

            n_components = int(np.rint(len(covariance)/2))

            pca = PCA(n_components=n_components)
            X_trans = pca.fit_transform(X) # pylint: disable=invalid-name

            kernela = KernelADASYN(proportion=self.proportion,
                              k=self.k,
                              nn_params=self.nn_params,
                              h=self.h,
                              random_state=self._random_state_init)

            X_samp, y_samp = kernela.sample(X_trans, y)

            return pca.inverse_transform(X_samp), y_samp

        samples = self.mcmc(X_min, r_score, covariance, n_to_sample)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

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

        X_min = X[y == self.min_label]

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting the nearest neighbors model
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=min([len(X_min), self.k+1]),
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        indices = nnmt.kneighbors(X_min, return_distance=False)

        # computing majority score
        r_score = np.array([np.sum(y[indices[i][1:]] == self.maj_label)
                      for i in range(len(X_min))])

        if np.sum(r_score > 0) < 2:
            return self.return_copies(X, y, "majority score is 0 for most "\
                                                "vectors")

        r_score = fix_density(r_score)

        # covariance is used to generate a random sample in the neighborhood
        covariance = cov(X_min[r_score > 0], rowvar=False)

        return self.iterated_sampling(covariance=covariance,
                                        X=X, y=y,
                                        X_min=X_min,
                                        r_score=r_score,
                                        n_to_sample=n_to_sample)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'k': self.k,
                'nn_params': self.nn_params,
                'h': self.h,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
