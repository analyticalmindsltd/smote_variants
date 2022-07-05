import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SN_SMOTE']

class SN_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @Article{sn_smote,
                        author="Garc{\'i}a, V.
                        and S{\'a}nchez, J. S.
                        and Mart{\'i}n-F{\'e}lez, R.
                        and Mollineda, R. A.",
                        title="Surrounding neighborhood-based SMOTE for
                                learning from imbalanced data sets",
                        journal="Progress in Artificial Intelligence",
                        year="2012",
                        month="Dec",
                        day="01",
                        volume="1",
                        number="4",
                        pages="347--362",
                        issn="2192-6360",
                        doi="10.1007/s13748-012-0027-5",
                        url="https://doi.org/10.1007/s13748-012-0027-5"
                        }
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                    to sample e.g. 1.0 means that after
                                    sampling the number of minority samples
                                    will be equal to the number of majority
                                    samples
            n_neighbors (float): number of neighbors
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
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
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
                                  'n_neighbors': [3, 5, 7]}
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

        X_min = X[y == self.min_label]

        # the search for the k nearest centroid neighbors is limited for the
        # nearest 10*n_neighbors neighbors
        
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)
        
        n_neighbors = min([self.n_neighbors*10, len(X_min)])
        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **nn_params)
        nn.fit(X_min)
        ind = nn.kneighbors(X_min, return_distance=False)

        # determining k nearest centroid neighbors
        ncn = np.zeros(shape=(len(X_min), self.n_neighbors)).astype(int)
        ncn_nums = np.zeros(len(X_min)).astype(int)

        metric_tensor= nn_params['metric_tensor'] if nn_params.get('metric_tensor', None) is not None else np.eye(len(X[0]))

        # extracting nearest centroid neighbors
        for i in range(len(X_min)):
            # the first NCN neighbor is the first neighbor
            ncn[i, 0] = ind[i][1]

            # iterating through all neighbors and finding the one with smaller
            # centroid distance to X_min[i] than the previous set of neighbors
            n_cent = 1
            centroid = X_min[ncn[i, 0]]
            #cent_dist = np.linalg.norm(centroid - X_min[i])
            cent_dist = np.sqrt(np.dot(np.dot((centroid - X_min[i]), metric_tensor), (centroid - X_min[i])))
            j = 2
            while j < len(ind[i]) and n_cent < self.n_neighbors:
                #new_cent_dist = np.linalg.norm(
                #    (centroid + X_min[ind[i][j]])/(n_cent + 1) - X_min[i])
                diff_vect = (centroid + X_min[ind[i][j]])/(n_cent + 1) - X_min[i]
                new_cent_dist = np.sqrt(np.dot(np.dot(diff_vect, metric_tensor), diff_vect))

                # checking if new nearest centroid neighbor found
                if new_cent_dist < cent_dist:
                    centroid = centroid + X_min[ind[i][j]]
                    ncn[i, n_cent] = ind[i][j]
                    n_cent = n_cent + 1
                    cent_dist = new_cent_dist
                j = j + 1

            # registering the number of nearest centroid neighbors found
            ncn_nums[i] = n_cent

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            random_idx = self.random_state.randint(len(X_min))
            random_neighbor_idx = self.random_state.choice(
                ncn[random_idx][:ncn_nums[random_idx]])
            samples.append(self.sample_between_points(
                X_min[random_idx], X_min[random_neighbor_idx]))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
