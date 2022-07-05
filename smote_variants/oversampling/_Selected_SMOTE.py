import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['Selected_SMOTE']


class Selected_SMOTE(OverSampling):
    """
    References:
        * BibTex::

        @article{smote_out_smote_cosine_selected_smote,
                  title={SMOTE-Out, SMOTE-Cosine, and Selected-SMOTE: An
                            enhancement strategy to handle imbalance in
                            data level},
                  author={Fajri Koto},
                  journal={2014 International Conference on Advanced
                            Computer Science and Information System},
                  year={2014},
                  pages={280-284}
                }

    Notes:
        * Significant attribute selection was not described in the paper,
            therefore we have implemented something meaningful.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 perc_sign_attr=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            strategy (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            n_neighbors (int): parameter of the NearestNeighbors component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            perc_sign_attr (float): [0,1] - percentage of significant
                                            attributes
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_neighbors, 'n_neighbors', 1)
        self.check_in_range(perc_sign_attr, 'perc_sign_attr', [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.perc_sign_attr = perc_sign_attr
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
                                  'n_neighbors': [3, 5, 7],
                                  'perc_sign_attr': [0.3, 0.5, 0.8]}
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

        if not self.check_enough_min_samples_for_sampling(3):
            return X.copy(), y.copy()

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])
        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]
        X_maj = X[y == self.maj_label]

        minority_indices = np.where(y == self.min_label)[0]

        n_neighbors = min([len(X_min), self.n_neighbors + 1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn_min_euc= NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                        n_jobs=self.n_jobs, 
                                                        **(nn_params))
        nn_min_euc.fit(X_min)

        nn_min_ind = nn_min_euc.kneighbors(X_min, return_distance=False)

        # significant attribute selection was not described in the paper
        # I have implemented significant attribute selection by checking
        # the overlap between ranges of minority and majority class attributes
        # the attributes with bigger overlap respecting their ranges
        # are considered more significant
        min_ranges_a = np.min(X_min, axis=0)
        min_ranges_b = np.max(X_min, axis=0)
        maj_ranges_a = np.min(X_maj, axis=0)
        maj_ranges_b = np.max(X_maj, axis=0)

        # end points of overlaps
        max_a = np.max(np.vstack([min_ranges_a, maj_ranges_a]), axis=0)
        min_b = np.min(np.vstack([min_ranges_b, maj_ranges_b]), axis=0)

        # size of overlap
        overlap = min_b - max_a

        # replacing negative values (no overlap) by zero
        overlap = np.where(overlap < 0, 0, overlap)
        # percentage of overlap compared to the ranges of attributes in the
        # minority set
        percentages = overlap/(min_ranges_b - min_ranges_a)
        # fixing zero division if some attributes have zero range
        percentages = np.nan_to_num(percentages)
        # number of significant attributes to determine
        num_sign_attr = min(
            [1, int(np.rint(self.perc_sign_attr*len(percentages)))])

        significant_attr = (percentages >= sorted(
            percentages)[-num_sign_attr]).astype(int)

        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.choice(range(len(minority_indices)))
            u = X[minority_indices[random_idx]]
            v = X_min[self.random_state.choice(nn_min_ind[random_idx][1:])]
            samples.append(self.sample_between_points_componentwise(
                u, v, significant_attr))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'perc_sign_attr': self.perc_sign_attr,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
