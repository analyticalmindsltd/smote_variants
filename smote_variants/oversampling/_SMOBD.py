import numpy as np

from sklearn.cluster import OPTICS

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['SMOBD']

class SMOBD(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{smobd,
                            author={Cao, Q. and Wang, S.},
                            booktitle={2011 International Conference on
                                        Information Management, Innovation
                                        Management and Industrial
                                        Engineering},
                            title={Applying Over-sampling Technique Based
                                     on Data Density and Cost-sensitive
                                     SVM to Imbalanced Learning},
                            year={2011},
                            volume={2},
                            number={},
                            pages={543-548},
                            keywords={data handling;learning (artificial
                                        intelligence);support vector machines;
                                        oversampling technique application;
                                        data density;cost sensitive SVM;
                                        imbalanced learning;SMOTE algorithm;
                                        data distribution;density information;
                                        Support vector machines;Classification
                                        algorithms;Noise measurement;Arrays;
                                        Noise;Algorithm design and analysis;
                                        Training;imbalanced learning;
                                        cost-sensitive SVM;SMOTE;data density;
                                        SMOBD},
                            doi={10.1109/ICIII.2011.276},
                            ISSN={2155-1456},
                            month={Nov},}
    """

    categories = [OverSampling.cat_uses_clustering,
                  OverSampling.cat_density_based,
                  OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 eta1=0.5,
                 t=1.8,
                 min_samples=5,
                 nn_params={},
                 max_eps=1.0,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            eta1 (float): control parameter of density estimation
            t (float): control parameter of noise filtering
            min_samples (int): minimum samples parameter for OPTICS
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            max_eps (float): maximum environment radius parameter for OPTICS
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_in_range(eta1, 'eta1', [0.0, 1.0])
        self.check_greater_or_equal(t, 't', 0)
        self.check_greater_or_equal(min_samples, 'min_samples', 1)
        self.check_greater_or_equal(max_eps, 'max_eps', 0.0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.eta1 = eta1
        self.t = t
        self.min_samples = min_samples
        self.nn_params = nn_params
        self.max_eps = max_eps
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
                                  'eta1': [0.1, 0.5, 0.9],
                                  't': [1.5, 2.5],
                                  'min_samples': [5],
                                  'max_eps': [0.1, 0.5, 1.0, 2.0]}
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

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # running the OPTICS technique based on the sklearn implementation
        min_samples = min([len(X_min)-1, self.min_samples])
        o = OPTICS(min_samples=min_samples,
                   max_eps=self.max_eps,
                   n_jobs=self.n_jobs)
        o.fit(X_min)
        cd = o.core_distances_
        rd = o.reachability_

        # noise filtering
        cd_average = np.mean(cd)
        rd_average = np.mean(rd)
        noise = np.logical_and(cd > cd_average*self.t, rd > rd_average*self.t)

        # fitting a nearest neighbor model to be able to find
        # neighbors in radius
        n_neighbors = min([len(X_min), self.min_samples+1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        indices = nn.kneighbors(X_min, return_distance=False)

        # determining the density
        factor_1 = cd
        factor_2 = np.array([len(x) for x in nn.radius_neighbors(
            X_min, radius=self.max_eps, return_distance=False)])

        if abs(max(factor_1)) < 1e-9 or abs(max(factor_2)) < 1e-9:
            return X.copy(), y.copy()

        if np.any(factor_1 != np.inf) or np.any(factor_2 != np.inf):
            return X.copy(), y.copy()

        factor_1[factor_1 == np.inf]= max(factor_1[factor_1 != np.inf])*1.1
        factor_2[factor_2 == np.inf]= max(factor_2[factor_2 != np.inf])*1.1

        factor_1 = factor_1/max(factor_1)
        factor_2 = factor_2/max(factor_2)

        df = factor_1*self.eta1 + factor_2*(1 - self.eta1)

        # setting the density at noisy samples to zero
        for i in range(len(noise)):
            if noise[i]:
                df[i] = 0

        if sum(df) == 0 or any(np.isnan(df)) or any(np.isinf(df)):
            return X.copy(), y.copy()

        # normalizing the density
        df_dens = df/sum(df)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min)), p=df_dens)
            neighbor_idx = self.random_state.choice(indices[idx][1:])
            samples.append(self.sample_between_points_componentwise(
                X_min[idx], X_min[neighbor_idx]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'eta1': self.eta1,
                't': self.t,
                'min_samples': self.min_samples,
                'nn_params': self.nn_params,
                'max_eps': self.max_eps,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
