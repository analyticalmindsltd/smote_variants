"""
This module implements the MDO method.
"""

import numpy as np

from sklearn.decomposition import PCA

from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSampling
from .._logger import logger
_logger= logger

__all__= ['MDO']

class MDO(OverSampling):
    """
    References:
        * BibTex::

            @ARTICLE{mdo,
                        author={Abdi, L. and Hashemi, S.},
                        journal={IEEE Transactions on Knowledge and Data
                                    Engineering},
                        title={To Combat Multi-Class Imbalanced Problems
                                by Means of Over-Sampling Techniques},
                        year={2016},
                        volume={28},
                        number={1},
                        pages={238-251},
                        keywords={covariance analysis;learning (artificial
                                    intelligence);modelling;pattern
                                    classification;sampling methods;
                                    statistical distributions;minority
                                    class instance modelling;probability
                                    contour;covariance structure;MDO;
                                    Mahalanobis distance-based oversampling
                                    technique;data-oriented technique;
                                    model-oriented solution;machine learning
                                    algorithm;data skewness;multiclass
                                    imbalanced problem;Mathematical model;
                                    Training;Accuracy;Eigenvalues and
                                    eigenfunctions;Machine learning
                                    algorithms;Algorithm design and analysis;
                                    Benchmark testing;Multi-class imbalance
                                    problems;over-sampling techniques;
                                    Mahalanobis distance;Multi-class imbalance
                                    problems;over-sampling techniques;
                                    Mahalanobis distance},
                        doi={10.1109/TKDE.2015.2458858},
                        ISSN={1041-4347},
                        month={Jan}}
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 K2=5,
                 K1_frac=0.5,
                 nn_params={},
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
            K2 (int): number of neighbors
            K1_frac (float): the fraction of K2 to set K1
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(K2, "K2", 1)
        self.check_greater_or_equal(K1_frac, "K1_frac", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.K2 = K2 # pylint: disable=invalid-name
        self.K1_frac = K1_frac # pylint: disable=invalid-name
        self.nn_params = nn_params
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
                                  'K2': [3, 5, 7],
                                  'K1_frac': [0.3, 0.5, 0.7]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def generate_1_sample(self,
                            alpha,
                            V # pylint: disable=invalid-name
                            ):
        """
        Generate 1 sample.

        Args:
            alpha (float): the alpha value
            V (np.array): the variances

        Returns:
            np.array: the generated sample
        """
        alpha_V = alpha * V # pylint: disable=invalid-name
        alpha_V[alpha_V < 0.001] = 0.001

        # initializing a new vector
        X_new = np.zeros(V.shape[0])

        # sampling components of the new vector
        s = 0# pylint: disable=invalid-name
        for jdx in range(V.shape[0] - 1):
            r = (2 * self.random_state.random_sample() - 1) # pylint: disable=invalid-name
            r = r * np.sqrt(alpha_V[jdx]) # pylint: disable=invalid-name
            X_new[jdx] = r
            s = s + (r**2 / alpha_V[jdx]) # pylint: disable=invalid-name

        last_fea_val = np.sqrt(max((1 - s) * alpha * V[-1], 0)) * (s <= 1)

        # determine last component to fulfill the ellipse equation
        X_new[-1] = (2 * self.random_state.random_sample() - 1) \
                                                        * last_fea_val

        return X_new

    def generate_samples(self, *, X_sel, weights, n_to_sample):
        """
        Generate samples.

        Args:
            X_sel (np.array): the selected samples
            weights (np.array): the weights
            n_to_sample (int): the number of samples

        Returns:
            np.array: the generated samples
        """
        # Algorithm 1 - MDO over-sampling
        mu = np.mean(X_sel, axis=0) # pylint: disable=invalid-name
        Z = X_sel - mu # pylint: disable=invalid-name
        # executing PCA
        pca = PCA(n_components=min([Z.shape[1], Z.shape[0]])).fit(Z)
        T = pca.transform(Z) # pylint: disable=invalid-name

        # computing variances (step 13)
        V = np.var(T, axis=0) # pylint: disable=invalid-name

        V[V < 0.001] = 0.001

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            # selecting a sample randomly according to the distribution
            idx = self.random_state.choice(np.arange(X_sel.shape[0]),
                                            p=weights)

            # finding vector in PCA space

            # computing alpha
            alpha = np.sum(T[idx]**2 / V)

            X_new = self.generate_1_sample(alpha, V)

            # append to new samples
            samples.append(X_new)

        return pca.inverse_transform(np.vstack(samples)) + mu

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

        # determining K1
        K1 = int(self.K2*self.K1_frac) # pylint: disable=invalid-name
        K1 = min([K1, len(X)]) # pylint: disable=invalid-name
        K2 = min([self.K2 + 1, len(X)]) # pylint: disable=invalid-name

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        # Algorithm 2 - chooseSamples
        nnmt = NearestNeighborsWithMetricTensor(n_neighbors=K2,
                                                n_jobs=self.n_jobs,
                                                **(nn_params))
        nnmt.fit(X)
        ind = nnmt.kneighbors(X_min, return_distance=False)

        # extracting the number of minority samples in local neighborhoods
        n_min = np.sum(y[ind[:, 1:]] == self.min_label, axis=1)

        # extracting selected samples from minority ones
        X_sel = X_min[n_min >= K1] # pylint: disable=invalid-name

        # falling back to returning input data if all the input is considered
        # noise
        if X_sel.shape[0] == 0:
            return self.return_copies(X, y, "No samples selected")

        # computing distribution
        weights = n_min[n_min >= K1] / K2
        weights = weights/np.sum(weights)

        samples = self.generate_samples(X_sel=X_sel,
                                        weights=weights,
                                        n_to_sample=n_to_sample)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'K2': self.K2,
                'K1_frac': self.K1_frac,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
