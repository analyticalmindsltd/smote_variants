import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['ANS']

class ANS(OverSampling):
    """
    References:
        * BibTex::

            @article{ans,
                     author = {Siriseriwan, W and Sinapiromsaran, Krung},
                     year = {2017},
                     month = {09},
                     pages = {565-576},
                     title = {Adaptive neighbor synthetic minority oversampling
                                technique under 1NN outcast handling},
                     volume = {39},
                     booktitle = {Songklanakarin Journal of Science and
                                    Technology}
                     }

    Notes:
        * The method is not prepared for the case when there is no c satisfying
            the condition in line 25 of the algorithm, fixed.
        * The method is not prepared for empty Pused sets, fixed.
    """
    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_ordinary,
                  OverSampling.cat_density_based,
                  OverSampling.cat_metric_learning]

    def __init__(self, 
                 proportion=1.0, 
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
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
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
        parameter_combinations = {'proportion': [
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]}
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

        # outcast extraction algorithm

        # maximum C value
        C_max = int(0.25*len(X))

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        # finding the first minority neighbor of minority samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=2, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min)
        dist, ind = nn.kneighbors(X_min)

        # extracting the distances of first minority neighbors from minority
        # samples
        first_pos_neighbor_distances = dist[:, 1]

        # fitting another nearest neighbors model to extract majority
        # samples in the neighborhoods of minority samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)

        # extracting the number of majority samples in the neighborhood of
        # minority samples
        out_border = []
        for i in range(len(X_min)):
            x = X_min[i].reshape(1, -1)
            ind = nn.radius_neighbors(x,
                                      first_pos_neighbor_distances[i],
                                      return_distance=False)
            #print('a', ind)
            out_border.append(np.sum(y[ind[0]] == self.maj_label))

        out_border = np.array(out_border)

        # finding the optimal C value by comparing the number of outcast
        # minority samples when traversing the range [1, C_max]
        n_oc_m1 = -1
        C = 0
        best_diff = np.inf
        for c in range(1, C_max):
            n_oc = np.sum(out_border >= c)
            if abs(n_oc - n_oc_m1) < best_diff:
                best_diff = abs(n_oc - n_oc_m1)
                C = n_oc
            n_oc_m1 = n_oc

        # determining the set of minority samples Pused
        Pused = np.where(out_border < C)[0]

        # Adaptive neighbor SMOTE algorithm

        # checking if there are minority samples left
        if len(Pused) == 0:
            _logger.info(self.__class__.__name__ + ": " + "Pused is empty")
            return X.copy(), y.copy()

        # finding the maximum distances of first positive neighbors
        eps = np.max(first_pos_neighbor_distances[Pused])

        # fitting nearest neighbors model to find nearest minority samples in
        # the neighborhoods of minority samples
        nn = NearestNeighborsWithMetricTensor(n_neighbors=1, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X_min[Pused])
        ind = nn.radius_neighbors(X_min[Pused], eps, return_distance=False)

        # extracting the number of positive samples in the neighborhoods
        Np = np.array([len(i) for i in ind])

        if np.all(Np == 1):
            message = "all samples have only 1 neighbor in the given radius"
            _logger.warning(self.__class__.__name__ + ": " + message)
            return X.copy(), y.copy()

        # determining the distribution used to generate samples
        distribution = Np/np.sum(Np)

        # generating samples
        samples = []
        while len(samples) < n_to_sample:
            random_idx = self.random_state.choice(
                np.arange(len(Pused)), p=distribution)
            if len(ind[random_idx]) > 1:
                random_neig_idx = self.random_state.choice(ind[random_idx])
                while random_neig_idx == random_idx:
                    random_neig_idx = self.random_state.choice(ind[random_idx])
                X_a = X_min[Pused[random_idx]]
                X_b = X_min[Pused[random_neig_idx]]
                samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}

