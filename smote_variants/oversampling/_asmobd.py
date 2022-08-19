"""
This module implements the ASMOBD method.
"""
import warnings

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS

from ..base import coalesce_dict, safe_divide, fix_density, coalesce
from ..base import NearestNeighborsWithMetricTensor
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['ASMOBD']

class ASMOBD(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_noise_removal,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 min_samples=3,
                 eps=0.8,
                 eta=0.5,
                 T_12=(1.0, 1.0),
                 t_12=(4.0, 4.0),
                 a=0.05,
                 smoothing='linear',
                 nn_params=None,
                 ss_params=None,
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
            min_samples (int): parameter of OPTICS
            eps (float): parameter of OPTICS
            eta (float): tradeoff parameter
            T_12 tuple(float): noise threshold (see paper)
            t_12 tuple(float): noise threshold (see paper)
            a (float): smoothing factor (see paper)
            smoothing (str): 'sigmoid'/'linear'
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params = coalesce_dict(ss_params, {'n_dim': 2,
                                            'simplex_sampling': 'uniform',
                                            'within_simplex_sampling': 'random',
                                            'gaussian_component': None})

        super().__init__(**ss_params, random_state=random_state)

        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(min_samples, "min_samples", 1)
        self.check_greater(eps, "eps", 0)
        self.check_in_range(eta, "eta", [0, 1])
        self.check_greater(a, "a", 0)
        self.check_isin(smoothing, "smoothing", ['sigmoid', 'linear'])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        # preventing too many instance attributes
        self.params = {'min_samples': min_samples,
                        'eps': eps,
                        'eta': eta,
                        'T_1': T_12[0],
                        'T_2': T_12[1],
                        't_1': t_12[0],
                        't_2': t_12[1],
                        'a': a,
                        'smoothing': smoothing}

        self.nn_params = coalesce(nn_params, {})
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
                                  'min_samples': [3],
                                  'eps': [0.3],
                                  'eta': [0.5],
                                  'T_12': [(0.7, 0.7), (0.7, 1.0), (0.7, 1.4),
                                            (1.0, 0.7), (1.0, 1.0), (1.0, 1.4),
                                            (1.4, 0.7), (1.4, 1.0), (1.4, 1.4)],
                                  't_12': [(4.0, 4.0)],
                                  'a': [0.05, 0.1],
                                  'smoothing': ['sigmoid', 'linear']}

        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def calculate_density(self, *, core_distances, ratio, n_min, n_maj, n_to_sample, params):
        """
        Calculate the density used for sampling base points.

        Args:
            core_distances (np.array): the core distances
            ratio (np.array): the n_maj/n_min ratios
            n_min (int): the n_min value
            n_maj (int): the n_maj values in the neighborhoods
            n_to_sample (int): number of samples to generate
            params (dict): the ASMOBD parameters

        Returns:
            np.array: the density for sampling
        """
        eta = params['eta']
        param_a = params['a']
        # calculating density according to the smoothing function specified
        if params['smoothing'] == 'sigmoid':
            balance_ratio = np.abs(2.0 / (1.0 + np.exp(-param_a * ratio)) - 1)
            density = eta * core_distances + (1.0 - eta)*n_maj
            density = density - balance_ratio
        else:
            density = eta * core_distances + (1.0 - eta)*n_maj
            density = eta * density + (1.0 - eta)*n_min/n_to_sample

        return density

    def determine_not_noise(self, core_distances, reachability, params):
        """
        Determine the not-noise mask

        Args:
            core_distances (np.array): the core distances
            reachability (np.array): the reachability values
            params (dict): the ASMOBD params

        Returns:
            np.array: the mask of non-noise instances
        """
        # identifying noise
        noise_1 = np.logical_and(core_distances > params['T_1'],
                                reachability > params['T_2'])

        # second constraint on noise
        mean_cd = np.mean(core_distances)
        mean_r = np.mean(reachability)
        noise_2 = np.logical_and(core_distances > mean_cd * params['t_1'],
                                reachability > mean_r * params['t_2'])

        # unifying the conditions on noise
        not_noise = np.logical_not(np.logical_or(noise_1, noise_2))

        return not_noise

    def determine_maj_min_ratios(self, *, X_min, core_distances, X_ss, y,
                                        nn_params, params):
        """
        Determine the majority/minority ratios.

        Args:
            X_min (np.array): the minority indices
            core_distances (np.array): core distances
            X_ss (np.array): the standardized base data
            y (np.array): the target labels
            nn_params (dict): the nearest neighbors parameters
            params (dict): the ASMOBD parameters

        Returns:
            np.array, np.array, np.array: the minority, majority counts
                                        and their ratios.
        """
        nearestn = NearestNeighborsWithMetricTensor(n_neighbors=params['min_samples'],
                                                    n_jobs=self.n_jobs,
                                                    **(nn_params))
        nearestn.fit(X_ss)

        n_majs = []
        n_mins = []
        ratio = []
        for idx, row in enumerate(X_min):
            ind = nearestn.radius_neighbors(row.reshape(1, -1),
                                            radius=core_distances[idx],
                                            return_distance=False)[0]
            n_majs.append(np.sum(y[ind.astype(int)] == self.maj_label)/len(ind))
            n_mins.append(len(ind) - n_majs[-1] - 1)
            ratio.append(safe_divide(n_majs[-1], n_mins[-1], np.inf))

        n_majs = np.array(n_majs)
        n_mins = np.array(n_mins)
        ratio = np.array(ratio)

        return n_mins, n_majs, ratio

    def determine_not_noise_and_density(self, *, X_ss, y, X_min,
                                        nn_params, n_to_sample, params):
        """
        Determine the not-noise mask and the density.

        Args:
            X_ss (np.array): the standardized base data
            y (np.array): the target labels
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbors parameters
            n_to_sample (int): the number of samples to generate
            params (dict): the ASMOBD parameters

        Returns:
            np.array, np.array: the mask of non-noise samples and the density
        """
        # executing the optics algorithm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optics = OPTICS(min_samples=\
                                np.min([len(X_min)-1, params['min_samples']]),
                            max_eps=params['eps'],
                            n_jobs=self.n_jobs)
            optics.fit(X_min)

        # fitting nearest neighbors models to identify the number of majority
        # samples in local environments
        n_min, n_maj, ratio = self.determine_maj_min_ratios(X_min=X_min,
                                                    core_distances=optics.core_distances_,
                                                    X_ss=X_ss,
                                                    y=y,
                                                    nn_params=nn_params,
                                                    params=params)

        not_noise = self.determine_not_noise(optics.core_distances_,
                                                optics.reachability_,
                                                params)
        density = self.calculate_density(core_distances=optics.core_distances_,
                                        ratio=ratio,
                                        n_min=n_min,
                                        n_maj=n_maj,
                                        n_to_sample=n_to_sample,
                                        params=params)

        return not_noise, density

    def determine_not_noise_and_density_safe(self, *, X_ss, y, X_min,
                                                nn_params, n_to_sample):
        """
        Determining the non-noise and density samples by releasing
        the conditions until some non-noise samples are detected.

        Args:
            X_ss (np.array): the standardized base data
            y (np.array): the target labels
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbors parameters
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array, np.array: the mask of non-noise samples and the density
        """
        multiplier = 1.5
        params_tmp = self.params.copy()

        for _ in range(4):
            not_noise, density = self.determine_not_noise_and_density(X_ss=X_ss,
                                                        y=y,
                                                        X_min=X_min,
                                                        nn_params=nn_params,
                                                        n_to_sample=n_to_sample,
                                                        params=params_tmp)
            if np.sum(not_noise) < 2:
                params_tmp['eps'] *= multiplier
                params_tmp['T_1'] *= multiplier
                params_tmp['T_2'] *= multiplier
                params_tmp['t_1'] *= multiplier
                params_tmp['t_2'] *= multiplier
            else:
                return not_noise, density

        return not_noise, density

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

        # standardizing the data to enable using absolute thresholds
        scaler = StandardScaler().fit(X)
        X_ss = scaler.transform(X) # pylint: disable=invalid-name

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X_ss, y)

        X_min = X_ss[y == self.min_label]

        not_noise, density = self.determine_not_noise_and_density_safe(X_ss=X_ss,
                                                    y=y,
                                                    X_min=X_min,
                                                    nn_params=nn_params,
                                                    n_to_sample=n_to_sample)
        # removing noise and adjusting the density factors accordingly
        X_min_not_noise = X_min[not_noise] # pylint: disable=invalid-name

        # checking if there are not-noisy samples
        if len(X_min_not_noise) <= 2:
            return self.return_copies(X, y, "not enough not-noise minority sample remained")

        density = np.delete(density, np.where(np.logical_not(not_noise))[0])
        density = fix_density(density)

        # fitting nearest neighbors model to non-noise minority samples
        n_neighbors = np.min([len(X_min_not_noise), self.params['min_samples'] + 1])
        nn_not_noise = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        **(nn_params))
        nn_not_noise.fit(X_min_not_noise)
        ind = nn_not_noise.kneighbors(X_min_not_noise, return_distance=False)

        samples = self.sample_simplex(X=X_min_not_noise,
                                        indices=ind,
                                        n_to_sample=n_to_sample,
                                        base_weights=density)

        return (np.vstack([X, scaler.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'min_samples': self.params['min_samples'],
                'eps': self.params['eps'],
                'eta': self.params['eta'],
                'T_12': (self.params['T_1'], self.params['T_2']),
                't_12': (self.params['t_1'], self.params['t_2']),
                'a': self.params['a'],
                'smoothing': self.params['smoothing'],
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
