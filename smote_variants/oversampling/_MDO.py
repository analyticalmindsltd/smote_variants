import numpy as np

from sklearn.decomposition import PCA

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
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
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(K2, "K2", 1)
        self.check_greater_or_equal(K1_frac, "K1_frac", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.K2 = K2
        self.K1_frac = K1_frac
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
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'K2': [3, 5, 7],
                                  'K1_frac': [0.3, 0.5, 0.7]}
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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # determining K1
        self.K1 = int(self.K2*self.K1_frac)
        K1 = min([self.K1, len(X)])
        K2 = min([self.K2 + 1, len(X)])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # Algorithm 2 - chooseSamples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=K2, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        ind = nn.kneighbors(X_min, return_distance=False)

        # extracting the number of minority samples in local neighborhoods
        n_min = np.array([np.sum(y[ind[i][1:]] == self.min_label)
                          for i in range(len(X_min))])

        # extracting selected samples from minority ones
        X_sel = X_min[n_min >= K1]

        # falling back to returning input data if all the input is considered
        # noise
        if len(X_sel) == 0:
            _logger.info(self.__class__.__name__ +
                         ": " + "No samples selected")
            return X.copy(), y.copy()

        # computing distribution
        weights = n_min[n_min >= K1]/K2
        weights = weights/np.sum(weights)

        # Algorithm 1 - MDO over-sampling
        mu = np.mean(X_sel, axis=0)
        Z = X_sel - mu
        # executing PCA
        pca = PCA(n_components=min([len(Z[0]), len(Z)])).fit(Z)
        T = pca.transform(Z)
        # computing variances (step 13)
        V = np.var(T, axis=0)

        V[V < 0.001] = 0.001

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            # selecting a sample randomly according to the distribution
            idx = self.random_state.choice(np.arange(len(X_sel)), p=weights)

            # finding vector in PCA space
            X_temp = T[idx]
            X_temp_square = X_temp**2

            # computing alphas
            alpha = np.sum(X_temp_square/V)
            alpha_V = alpha*V
            alpha_V[alpha_V < 0.001] = 0.001

            # initializing a new vector
            X_new = np.zeros(len(X_temp))

            # sampling components of the new vector
            s = 0
            for j in range(len(X_temp)-1):
                r = (2*self.random_state.random_sample()-1)*np.sqrt(alpha_V[j])
                X_new[j] = r
                s = s + (r**2/alpha_V[j])

            if s > 1:
                last_fea_val = 0
            else:
                tmp = (1 - s)*alpha*V[-1]
                if tmp < 0:
                    tmp = 0
                last_fea_val = np.sqrt(tmp)
            # determine last component to fulfill the ellipse equation
            X_new[-1] = (2*self.random_state.random_sample()-1)*last_fea_val
            # append to new samples
            samples.append(X_new)

        return (np.vstack([X, pca.inverse_transform(samples) + mu]),
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
                'random_state': self._random_state_init}
