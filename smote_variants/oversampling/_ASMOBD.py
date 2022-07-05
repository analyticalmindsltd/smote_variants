import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['ASMOBD']

class ASMOBD(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{asmobd,
                            author={Senzhang Wang and Zhoujun Li and Wenhan
                                    Chao and Qinghua Cao},
                            booktitle={The 2012 International Joint Conference
                                        on Neural Networks (IJCNN)},
                            title={Applying adaptive over-sampling technique
                                    based on data density and cost-sensitive
                                    SVM to imbalanced learning},
                            year={2012},
                            volume={},
                            number={},
                            pages={1-8},
                            doi={10.1109/IJCNN.2012.6252696},
                            ISSN={2161-4407},
                            month={June}}

    Notes:
        * In order to use absolute thresholds, the data is standardized.
        * The technique has many parameters, not easy to find the right
            combination.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_noise_removal,
                  OverSampling.cat_uses_clustering,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 min_samples=3,
                 eps=0.8,
                 eta=0.5,
                 T_1=1.0,
                 T_2=1.0,
                 t_1=4.0,
                 t_2=4.0,
                 a=0.05,
                 smoothing='linear',
                 iteration=0,
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
            min_samples (int): parameter of OPTICS
            eps (float): parameter of OPTICS
            eta (float): tradeoff parameter
            T_1 (float): noise threshold (see paper)
            T_2 (float): noise threshold (see paper)
            t_1 (float): noise threshold (see paper)
            t_2 (float): noise threshold (see paper)
            a (float): smoothing factor (see paper)
            smoothing (str): 'sigmoid'/'linear'
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
        self.check_greater_or_equal(min_samples, "min_samples", 1)
        self.check_greater(eps, "eps", 0)
        self.check_in_range(eta, "eta", [0, 1])
        self.check_greater(T_1, "T_1", 0)
        self.check_greater(T_2, "T_2", 0)
        self.check_greater(t_1, "t_1", 0)
        self.check_greater(t_2, "t_2", 0)
        self.check_greater(a, "a", 0)
        self.check_isin(smoothing, "smoothing", ['sigmoid', 'linear'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.min_samples = min_samples
        self.eps = eps
        self.eta = eta
        self.T_1 = T_1
        self.T_2 = T_2
        self.t_1 = t_1
        self.t_2 = t_2
        self.a = a
        self.smoothing = smoothing
        self.iteration = iteration
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
                                  'min_samples': [3],
                                  'eps': [0.3],
                                  'eta': [0.5],
                                  'T_1': [0.7, 1.0, 1.4],
                                  'T_2': [0.7, 1.0, 1.4],
                                  't_1': [4.0],
                                  't_2': [4.0],
                                  'a': [0.05, 0.1],
                                  'smoothing': ['sigmoid', 'linear']}

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

        # standardizing the data to enable using absolute thresholds
        ss = StandardScaler().fit(X)
        X_ss = ss.transform(X)

        X_min = X_ss[y == self.min_label]

        # executing the optics algorithm
        min_samples = min([len(X_min)-1, self.min_samples])
        o = OPTICS(min_samples=min_samples,
                   max_eps=self.eps,
                   n_jobs=self.n_jobs)
        o.fit(X_min)
        cd = o.core_distances_
        r = o.reachability_

        # identifying noise
        noise = np.logical_and(cd > self.T_1, r > self.T_2)

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # fitting nearest neighbors models to identify the number of majority
        # samples in local environments
        nn = NearestNeighborsWithMetricTensor(n_neighbors=self.min_samples, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_ss)
        n_majs = []
        ratio = []
        for i in range(len(X_min)):
            ind = nn.radius_neighbors(X_min[i].reshape(
                1, -1), radius=cd[i], return_distance=False)[0]
            n_maj = np.sum(y[ind] == self.maj_label)/len(ind)
            n_majs.append(n_maj)
            n_min = len(ind) - n_maj - 1
            if n_min == 0:
                ratio.append(np.inf)
            else:
                ratio.append(n_maj/n_min)

        n_maj = np.array(n_maj)
        ratio = np.array(ratio)

        # second constraint on noise
        noise_2 = np.logical_and(cd > np.mean(
            cd)*self.t_1, r > np.mean(r)*self.t_2)

        # calculating density according to the smoothing function specified
        if self.smoothing == 'sigmoid':
            balance_ratio = np.abs(2.0/(1.0 + np.exp(-self.a*ratio[i])) - 1.0)
            df = self.eta*cd + (1.0 - self.eta)*n_maj - balance_ratio
        else:
            df = self.eta*(self.eta*cd + (1.0 - self.eta)*n_maj) + \
                (1 - self.eta)*len(X_min)/n_to_sample

        # unifying the conditions on noise
        not_noise = np.logical_not(np.logical_or(noise, noise_2))

        # checking if there are not noise samples remaining
        if np.sum(not_noise) == 0:
            message = ("All minority samples found to be noise, increasing"
                       "noise thresholds")
            _logger.info(self.__class__.__name__ + ": " + message)
            
            if self.iteration == 3:
                _logger.warning(self.__class__.__name__ + ': ' + 'recursion is stopped, returning original dataset')
                return X.copy(), y.copy()

            return ASMOBD(proportion=self.proportion,
                          min_samples=self.min_samples,
                          eps=self.eps,
                          eta=self.eta,
                          T_1=self.T_1*1.5,
                          T_2=self.T_2*1.5,
                          t_1=self.t_1*1.5,
                          t_2=self.t_2*1.5,
                          a=self.a,
                          smoothing=self.smoothing,
                          iteration=self.iteration+1,
                          nn_params=nn_params,
                          n_jobs=self.n_jobs,
                          random_state=self._random_state_init).sample(X, y)

        # removing noise and adjusting the density factors accordingly
        X_min_not_noise = X_min[not_noise]

        # checking if there are not-noisy samples
        if len(X_min_not_noise) <= 2:
            _logger.warning(self.__class__.__name__ + ": " +
                            "no not-noise minority sample remained")
            return X.copy(), y.copy()

        df = np.nan_to_num(df, np.max(df[np.isfinite(df)])*1.1, posinf=True)

        df = np.delete(df, np.where(np.logical_not(not_noise))[0])

        if np.sum(df) == 0:
            df= np.repeat(1.0, len(df))

        density = df/np.sum(df)
        
        if np.any(np.isnan(density)):
            density= np.nan_to_num(density, nan=0.0)
            if not np.sum(density) == 0.0:
                density= density/np.sum(density)
            else:
                density= np.repeat(1.0/len(density), len(density))

        # fitting nearest neighbors model to non-noise minority samples
        n_neighbors = min([len(X_min_not_noise), self.min_samples + 1])
        nn_not_noise = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params))
        nn_not_noise.fit(X_min_not_noise)
        ind = nn_not_noise.kneighbors(X_min_not_noise, return_distance=False)

        # do the sampling
        samples = []
        while len(samples) < n_to_sample:
            idx = self.random_state.choice(np.arange(len(X_min_not_noise)),
                                           p=density)
            random_neighbor_idx = self.random_state.choice(ind[idx][1:])
            X_a = X_min_not_noise[idx]
            X_b = X_min_not_noise[random_neighbor_idx]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, ss.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'min_samples': self.min_samples,
                'eps': self.eps,
                'eta': self.eta,
                'T_1': self.T_1,
                'T_2': self.T_2,
                't_1': self.t_1,
                't_2': self.t_2,
                'a': self.a,
                'smoothing': self.smoothing,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
