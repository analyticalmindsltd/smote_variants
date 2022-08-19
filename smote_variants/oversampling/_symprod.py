"""
This module implements the SYMPROD method.
"""

import itertools

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..base import coalesce
from ..base import OverSampling
from ..base import (NearestNeighborsWithMetricTensor)

from .._logger import logger

__all__ = ['SYMPROD']

class SYMPROD(OverSampling):
    """
    References:
        * Bibtex::

            @article{kunakorntum2020synthetic,
                    title={A Synthetic Minority Based on Probabilistic
                            Distribution (SyMProD) Oversampling for Imbalanced
                            Datasets},
                    author={Kunakorntum, Intouch and Hinthong,
                            Woranich and Phunchongharn, Phond},
                    journal={IEEE Access},
                    volume={8},
                    pages={114692--114704},
                    year={2020},
                    publisher={IEEE}
                }
    """
    categories = [OverSampling.cat_noise_removal,
                  OverSampling.cat_density_based,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                    *,
                    proportion=1.0,
                    std_outliers=3,
                    k_neighbors=7,
                    m_neighbors=7,
                    cutoff_threshold=1.25,
                    nn_params=None,
                    n_jobs=1,
                    random_state=None,
                    **_kwargs):
        """
        Constructor of the SYMPROD sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            std_outliers (int): value for removing outliers based on standard
                                deviation of each point
            k_neighbors (int): number of nearest neighbors for calculating distance,
                                closeness factor of each point
            m_neighbors (int): number of nearest neighbors for generating synthetic
                                instances of each point.
            cutoff_threshold (float): threshold for removing minority points where
                                        locating in majority region
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
        self.check_greater_or_equal(std_outliers, "std_outliers", 1)
        self.check_greater_or_equal(k_neighbors, "k_neighbors", 1)
        self.check_greater_or_equal(m_neighbors, "m_neighbors", 1)
        self.check_greater_or_equal(cutoff_threshold, "cutoff_threshold", 0.01)

        self.proportion = proportion
        self.params = {'std_outliers': std_outliers,
                        'k_neighbors': k_neighbors,
                        'm_neighbors': m_neighbors,
                        'cutoff_threshold': cutoff_threshold}
        self.nn_params = coalesce(nn_params, {})
        self.n_jobs = n_jobs

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable paramter combinations.
        Returns:
            list(dict): a list of meaningful paramter combinations
        """
        parameter_combinations = {'proportion': [1.0],
                                  'std_outliers': [3, 4],
                                  'k_neighbors': [5, 7],
                                  'm_neighbors': [5, 7],
                                  'cutoff_threshold': [1.0, 1.25, 1.5]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def noise_removal(self, X, y, scaler):
        """
        Noise removal.

        Args:
            X (np.array): all training vectors
            y (np.array): all target labels
            scaler (obj): the scaler to be used

        Returns:
            np.array, np.array, np.array, np.array: the minority and majority
                            vectors after noise removal, and all training
                            vectors and labels after noise removal
        """
        def remove_outliers(X):
            return X[np.max(np.abs(scaler.fit_transform(X)), axis=1) \
                                            < self.params['std_outliers']]

        X_min = remove_outliers(X[y == self.min_label])
        X_maj = remove_outliers(X[y == self.maj_label])

        # Apply the original dataset if the dataset is null
        X_min = X[y == self.min_label] if len(X_min) == 0 else X_min
        X_maj = X[y == self.maj_label] if len(X_min) == 0 else X_maj

        # Create X, y after Noise Removal
        # seemingly the purpose of this is to standardize all data together
        y_nr= np.hstack([np.repeat(self.min_label, len(X_min)),
                        np.repeat(self.maj_label, len(X_maj))])
        X_nr= scaler.fit_transform(np.vstack([X_min, X_maj])) # pylint: disable=invalid-name

        # these X_min and X_maj are standardized now
        X_min = X_nr[y_nr == self.min_label]
        X_maj = X_nr[y_nr == self.maj_label]

        return X_min, X_maj, X_nr, y_nr

    def check_all_neighbors_the_same(self, dist, name=''):
        """
        Checks if all neighbors are the same.

        Args:
            dist (np.array): neighbor distances
            name (str): name of the neighborhood
        """
        if np.any(np.nansum(dist, axis=1) == 0.0):
            raise ValueError(f"all {name} samples are the same")

    def normalize(self, array):
        """
        Normalize the array.

        Args:
            array (np.array): the array to normalize

        Returns:
            np.array: the normalized array
        """
        if (array.max() - array.min()) == 0.0:
            array = np.repeat(0.0, len(array))
        else:
            array = (array - array.min()) / (array.max() - array.min())

        return array

    def determine_cf_norms(self, X_maj, X_min, nn_params):
        """
        Determine the closeness factors.

        Args:
            X_maj (np.array): the majority vectors
            X_min (np.array): the minority vectors
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array, np.array, np.array: the minority and majority
                                        closeness factors and the
                                        minority-minority neighborhood
                                        structure and neighbor distances
        """
        nn_maj = NearestNeighborsWithMetricTensor(n_neighbors=len(X_maj),
                                                    n_jobs=self.n_jobs,
                                                    **nn_params).fit(X_maj)
        d_maj_maj, _ = nn_maj.kneighbors(X_maj, return_distance=True)

        nn_min = NearestNeighborsWithMetricTensor(n_neighbors=len(X_min),
                                                    n_jobs=self.n_jobs,
                                                    **nn_params).fit(X_min)
        d_min_min, i_min_min = nn_min.kneighbors(X_min, return_distance=True)

        # Calculate Closeness Factor(CF)
        self.check_all_neighbors_the_same(d_min_min, 'minority')

        cf_min = 1.0 / np.nansum(d_min_min, axis=1)
        cf_norm_min = self.normalize(cf_min)

        self.check_all_neighbors_the_same(d_maj_maj, 'majority')

        cf_maj = 1.0 / np.nansum(d_maj_maj, axis=1)
        cf_norm_maj = self.normalize(cf_maj)

        return cf_norm_min, cf_norm_maj, i_min_min, d_min_min

    def determine_taus(self, X_maj, X_min, nn_params):
        """
        Determine the tau arrays.

        Args:
            X_maj (np.array): the majority vectors
            X_min (np.array): the minority vectors
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array, np.array: the minority and majority taus
                                            and the minority closeness factors
        """
        cf_norm_min, cf_norm_maj, i_min_min, d_min_min = \
            self.determine_cf_norms(X_maj, X_min, nn_params)

        # First, Calculate the distance from minority class to minority class for comparing
        # with majority class

        k_neighbors= np.min([self.params['k_neighbors'], len(X_min) - 1])

        cf_norm_min_idx = np.take(cf_norm_min, i_min_min[:, 1:(k_neighbors + 1)])
        tau_min = np.nanmean(cf_norm_min_idx/(d_min_min[:, 1:(k_neighbors + 1)] + 1),
                             axis=1)

        # Calculate distance from minority to majority
        nn_min_maj = NearestNeighborsWithMetricTensor(n_neighbors=k_neighbors,
                                                        n_jobs=self.n_jobs,
                                                        **nn_params).fit(X_maj)
        d_min_maj, i_min_maj = nn_min_maj.kneighbors(X_min, return_distance=True)

        tau_maj = (np.take(cf_norm_maj, i_min_maj) / (d_min_maj + 1)).mean(axis=1)

        return tau_min, tau_maj, cf_norm_min

    def check_nans(self, array, name=''):
        """
        Checks if NaNs are present in an array.

        Args:
            array (np.array): an array
            name (str): name of the array
        """

        if np.any(np.isnan(array)):
            raise ValueError(f"NaN values in the {name} array")

    def minority_point_selection(self, X_maj, X_min, nn_params):
        """
        Minority point selection.

        Args:
            X_maj (np.array): the majority vectors
            X_min (np.array): the minority vectors
            nn_params (dict): the nearest neighbor parameters

        Returns:
            np.array, np.array, np.array: the probability distribution,
                                        the minority closeness factors and
                                        the selected minority vectors
        """
        tau_min, tau_maj, cf_norm_min = self.determine_taus(X_maj, X_min, nn_params)

        # a cutoff threshold is needed which keeps at least 2 minority samples
        cutoff_threshold = self.params['cutoff_threshold']
        while np.sum(tau_min >= tau_maj * cutoff_threshold) <= 1:
            cutoff_threshold = cutoff_threshold - self.params['cutoff_threshold']/10.0

        logger.info("%s: Cutoff value updated from %f to %f",
                        self.__class__.__name__,
                        self.params['cutoff_threshold'], cutoff_threshold)

        mask = (tau_min >= tau_maj * cutoff_threshold)

        X_min = X_min[mask]
        cf_norm_min = cf_norm_min[mask]

        phi = (tau_min[mask] + 1)/(tau_maj[mask] + 1)
        phi = phi - phi.min()
        prob_dist = phi/phi.sum()

        self.check_nans(prob_dist, 'phi')

        return prob_dist, cf_norm_min, X_min

    def check_enough_neighbors(self, m_neighbors):
        """
        Checks if the number of samples is enough for the m_neighbors parameter

        Args:
            m_neighbors (int): the number of neighbors
        """
        if m_neighbors < 2:
            raise ValueError("Not enough samples for "\
                                f"m_neighbors {self.params['m_neighbors']}")

    def select_neighborhoods(self, X_min, nn_params, n_to_sample):
        """
        Select the neighborhoods used for sampling.

        Args:
            X_min (np.array): the minority samples
            nn_params (dict): the nearest neighbor parameters
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the selected neighbors
        """
        m_neighbors = np.min([self.params['m_neighbors'], len(X_min) - 4])

        self.check_enough_neighbors(m_neighbors)

        nn_min_min = NearestNeighborsWithMetricTensor(m_neighbors + 3,
                                                        n_jobs=self.n_jobs,
                                                        **nn_params).fit(X_min)
        _, i_min_min = nn_min_min.kneighbors(X_min, return_distance=True)

        # it is hardly understandable why M + 3 closest neighbors are selected
        # to take a random sample from
        neighborhoods = np.array(list(itertools.combinations(np.arange(m_neighbors + 3),
                                                    m_neighbors)))

        rand_ind = neighborhoods[self.random_state.choice(len(neighborhoods),
                                                          n_to_sample)]
        nn_selected = np.take(i_min_min, rand_ind)

        return nn_selected

    def instance_synthesis(self,
                            *,
                            prob_dist,
                            n_to_sample,
                            X_min,
                            nn_params,
                            cf_norm_min):
        """
        Instance synthesis.

        Args:
            prob_dist (np.array): the probability distribution
            n_to_sample (int): the number of samples to generate
            X_min (np.array): the minority vectors
            nn_params (dict): the nearest neighbor parameters
            cf_norm_min (np.array): the minority closeness factors

        Returns:
            np.array: the generated samples
        """
        min_reference = self.random_state.choice(len(prob_dist),
                                                    n_to_sample,
                                                    p=prob_dist,
                                                    replace=True)

        nn_selected = self.select_neighborhoods(X_min, nn_params, n_to_sample)

        synthetic_points = np.zeros(shape=(n_to_sample, X_min[0].shape[0]))

        feature_value = np.concatenate([X_min[nn_selected],
                                        X_min[min_reference][:, None, :]],
                                        axis=1)
        CF = np.hstack([cf_norm_min[nn_selected], # pylint: disable=invalid-name
                        cf_norm_min[min_reference][:, None]])
        dirichlet_param = np.ones(self.params['m_neighbors'] + 1) \
                                    * (self.params['m_neighbors'] + 2)
        random_beta = self.random_state.dirichlet(dirichlet_param,
                                                  size=min_reference.shape[0])

        calculate_CF = random_beta * CF # pylint: disable=invalid-name
        calculate_CF = (calculate_CF.T / np.sum(calculate_CF, axis=1)).T # pylint: disable=invalid-name

        feature_value = feature_value * calculate_CF[:, :, None]

        synthetic_points = np.sum(feature_value, axis=1)

        return synthetic_points

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

        ####################
        # I. Noise Removal #
        ####################

        scaler = StandardScaler()

        X_min, X_maj, X_nr, y_nr = self.noise_removal(X, y, scaler) # pylint: disable=invalid-name

        ################################
        # II. Minority point selection #
        ################################

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = \
                self.metric_tensor_from_nn_params(nn_params, X_nr, y_nr)

        try:
            prob_dist, cf_norm_min, X_min = \
                self.minority_point_selection(X_maj, X_min, nn_params)
        except ValueError as valueerror:
            return self.return_copies(X, y, valueerror.args[0])

        ###########################
        # III. Instance Synthesis #
        ###########################

        try:
            synthetic_points = self.instance_synthesis(prob_dist=prob_dist,
                                                    n_to_sample=n_to_sample,
                                                    X_min=X_min,
                                                    nn_params=nn_params,
                                                    cf_norm_min=cf_norm_min)
        except ValueError as valueerror:
            return self.return_copies(X, y, valueerror.args[0])

        return (np.vstack([X, np.vstack(scaler.inverse_transform(synthetic_points))]),
                np.hstack([y, np.repeat(self.min_label, len(synthetic_points))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'std_outliers': self.params['std_outliers'],
                'k_neighbors': self.params['k_neighbors'],
                'm_neighbors': self.params['m_neighbors'],
                'cutoff_threshold': self.params['cutoff_threshold'],
                'nn_params': self.nn_params,
                **OverSampling.get_params(self)}
