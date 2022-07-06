import numpy as np
import scipy.spatial as sspatial
from sklearn.preprocessing import StandardScaler

from ._OverSampling import OverSampling
from .._metric_tensor import (n_neighbors_func, MetricTensor, 
                                NearestNeighborsWithMetricTensor)

from .._logger import logger

__all__ = ['SYMPROD']

class SYMPROD(OverSampling):
    """
    References:
        * Bibtex::

            @article{kunakorntum2020synthetic,
                    title={A Synthetic Minority Based on Probabilistic Distribution (SyMProD) Oversampling for Imbalanced Datasets},
                    author={Kunakorntum, Intouch and Hinthong, Woranich and Phunchongharn, Phond},
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
                 proportion=1.0,
                 std_outliers=3,
                 k_neighbors=7,
                 m_neighbors=7,
                 cutoff_threshold=1.25,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
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
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(std_outliers, "std_outliers", 1)
        self.check_greater_or_equal(k_neighbors, "k_neighbors", 1)
        self.check_greater_or_equal(m_neighbors, "m_neighbors", 1)
        self.check_greater_or_equal(cutoff_threshold, "cutoff_threshold", 0.01)

        self.proportion = proportion
        self.std_outliers = std_outliers
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.cutoff_threshold = cutoff_threshold
        self.nn_params = nn_params
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

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

    def sample(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        self.class_label_statistics(X, y)

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            logger.warning(self.__class__.__name__ +
                           ": " + "Sampling is not needed")
            return X.copy(), y.copy()
        
        if self.class_stats[self.min_label] < 2:
            logger.warning(self.__class__.__name__ +
                            ": " + "not enough minority samples for oversampling: %d"\
                                 % self.class_stats[self.min_label])
            return X.copy(), y.copy()

        ####################
        # I. Noise Removal #
        ####################

        ss = StandardScaler()

        def remove_outliers(X):
            return X[np.max(np.abs(ss.fit_transform(X)), axis=1) < self.std_outliers]

        X_min = remove_outliers(X[y == self.min_label])
        X_maj = remove_outliers(X[y == self.maj_label])

        # Apply the original dataset if the dataset is null
        X_min = X[y == self.min_label] if len(X_min) == 0 else X_min
        X_maj = X[y == self.maj_label] if len(X_min) == 0 else X_maj

        # Create X, y after Noise Removal
        # seemingly the purpose of this is to standardize all data together
        y_nr= np.hstack([np.repeat(self.min_label, len(X_min)),
                        np.repeat(self.maj_label, len(X_maj))])
        X_nr= ss.fit_transform(np.vstack([X_min, X_maj]))

        # these X_min and X_maj are standardized now
        X_min = X_nr[y_nr == self.min_label]
        X_maj = X_nr[y_nr == self.maj_label]

        ################################
        # II. Minority point selection #
        ################################

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X_nr, y_nr)

        nn_maj = NearestNeighborsWithMetricTensor(n_neighbors=len(X_maj), 
                                                    n_jobs=self.n_jobs,
                                                    **nn_params).fit(X_maj)
        d_maj_maj, _ = nn_maj.kneighbors(X_maj, return_distance=True)

        nn_min = NearestNeighborsWithMetricTensor(n_neighbors=len(X_min),
                                                    n_jobs=self.n_jobs,
                                                    **nn_params).fit(X_min)
        d_min_min, i_min_min = nn_min.kneighbors(X_min, return_distance=True)

        # Calculate Closeness Factor(CF)
        if np.any(np.nansum(d_min_min, axis=1) == 0.0):
            logger.warning(self.__class__.__name__ + ": all minority samples are the same")
            return X.copy(), y.copy()
        
        cf_min = 1.0/np.nansum(d_min_min, axis=1)
        cf_norm_min = (cf_min - cf_min.min()) / (cf_min.max()-cf_min.min())
        cf_norm_min = np.nan_to_num(cf_norm_min, 0.5)

        if np.any(np.nansum(d_maj_maj, axis=1) == 0.0):
            logger.warning(self.__class__.__name__ + ": all majority samples are the same")
            return X.copy(), y.copy()

        cf_maj = 1.0/np.nansum(d_maj_maj, axis=1)
        cf_norm_maj = (cf_maj - cf_maj.min()) / (cf_maj.max()-cf_maj.min())
        cf_norm_maj = np.nan_to_num(cf_norm_maj, 0.5)

        # First, Calculate the distance from minority class to minority class for comparing 
        # with majority class

        k_neighbors= np.min([self.k_neighbors, len(X_min) - 1])

        cf_norm_min_idx = np.take(cf_norm_min, i_min_min[:, 1:(k_neighbors + 1)])
        tau_min = np.nanmean(cf_norm_min_idx/(d_min_min[:, 1:(k_neighbors + 1)] + 1), axis=1)

        # Calculate distance from minority to majority
        nn_min_maj = NearestNeighborsWithMetricTensor(n_neighbors=k_neighbors,
                                                        n_jobs=self.n_jobs, 
                                                        **nn_params).fit(X_maj)
        d_min_maj, i_min_maj = nn_min_maj.kneighbors(X_min, return_distance=True)

        cf_norm_maj_index = np.take(cf_norm_maj, i_min_maj)
        tau_maj = (cf_norm_maj_index / (d_min_maj+1)).mean(axis=1)

        # a cutoff threshold is needed which keeps at least 2 minority samples
        cutoff_threshold = self.cutoff_threshold
        while not np.sum(tau_min >= tau_maj * cutoff_threshold) > 1:
            cutoff_threshold = cutoff_threshold - self.cutoff_threshold/10.0

        logger.info(self.__class__.__name__ + \
            ": Cutoff value updated from %f to %f" % (self.cutoff_threshold, cutoff_threshold))

        mask = (tau_min >= tau_maj * cutoff_threshold)

        X_min = X_min[mask]
        cf_norm_min = cf_norm_min[mask]

        phi = (tau_min[mask] + 1)/(tau_maj[mask] + 1)
        phi = phi - phi.min()
        prob_dist = phi/phi.sum()

        if np.any(np.isnan(prob_dist)):
            logger.info(self.__class__.__name__ + \
                ": NaN values in the phi probability distribution, returning the original dataset")
            return X.copy(), y.copy()

        ###########################
        # III. Instance Synthesis #
        ###########################

        min_reference = self.random_state.choice(len(prob_dist),
                                                    n_to_sample, 
                                                    p=prob_dist, 
                                                    replace=True)
        
        m_neighbors = np.min([self.m_neighbors, len(X_min) - 4])

        if m_neighbors < 2:
            logger.info(self.__class__.__name__ + \
                ": Not enough samples, for the parameter m_neighbors: %d" % self.m_neighbors)
            return X.copy(), y.copy()

        nn_min_min = NearestNeighborsWithMetricTensor(m_neighbors + 3, 
                                                        n_jobs=self.n_jobs,
                                                        **nn_params).fit(X_min)
        d_min_min, i_min_min = nn_min_min.kneighbors(X_min, return_distance=True)
        
        # it is hardly understandable why M + 3 closest neighbors are selected
        # to take a random sample from
        rand_ind = [self.random_state.choice(len(i_min_min[0]), 
                                            size=m_neighbors, 
                                            replace=False) for _ in range(len(i_min_min))]
        nn_selected = np.take(i_min_min, rand_ind)

        synthetic_points = np.zeros(shape=(n_to_sample, X[0].shape[0]))

        # generating the samples
        for ind, point in enumerate(min_reference):
            # Merge feature value of neighbors and reference point
            feature_value = np.vstack([X_min[nn_selected[point]], X_min[[point]]])

            # closeness factor of reference and neighbors stacked
            CF = np.hstack([cf_norm_min[nn_selected[point]], cf_norm_min[[point]]])

            # Random value as beta
            dirichlet_param = np.ones(self.m_neighbors+1)*(self.m_neighbors+2)
            random_beta = self.random_state.dirichlet(dirichlet_param, size=1)
            
            calculate_CF = random_beta.T * CF[:,None]

            feature_value = feature_value * calculate_CF / calculate_CF.sum()
            
            synthetic_points[ind] = np.sum(feature_value, axis=0)

        return (np.vstack([X, np.vstack(ss.inverse_transform(synthetic_points))]),
                np.hstack([y, np.repeat(self.min_label, len(synthetic_points))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'std_outliers': self.std_outliers,
                'k_neighbors': self.k_neighbors,
                'm_neighbors': self.m_neighbors,
                'cutoff_threshold': self.cutoff_threshold,
                'nn_params': self.nn_params,
                'random_state': self._random_state_init}
