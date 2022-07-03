import numpy as np

from sklearn.metrics import pairwise_distances

from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SOMO']

class SOMO(OverSampling):
    """
    References:
        * BibTex::

            @article{somo,
                        title = "Self-Organizing Map Oversampling (SOMO) for
                                    imbalanced data set learning",
                        journal = "Expert Systems with Applications",
                        volume = "82",
                        pages = "40 - 52",
                        year = "2017",
                        issn = "0957-4174",
                        doi = "https://doi.org/10.1016/j.eswa.2017.03.073",
                        author = "Georgios Douzas and Fernando Bacao"
                        }

    Notes:
        * It is not specified how to handle those cases when a cluster contains
            1 minority samples, the mean of within-cluster distances is set to
            100 in these cases.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_uses_clustering]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_grid=10,
                 sigma=0.2,
                 learning_rate=0.5,
                 n_iter=100,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            n_grid (int): size of grid
            sigma (float): sigma of SOM
            learning_rate (float) learning rate of SOM
            n_iter (int): number of iterations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_grid, 'n_grid', 2)
        self.check_greater(sigma, 'sigma', 0)
        self.check_greater(learning_rate, 'learning_rate', 0)
        self.check_greater_or_equal(n_iter, 'n_iter', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_grid = n_grid
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iter = n_iter
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
                                  'n_grid': [5, 9, 13],
                                  'sigma': [0.4],
                                  'learning_rate': [0.3, 0.5],
                                  'n_iter': [100]}
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

        N_inter = n_to_sample/2
        N_intra = n_to_sample/2

        import minisom

        # training SOM
        som = minisom.MiniSom(self.n_grid,
                              self.n_grid,
                              len(X[0]),
                              sigma=self.sigma,
                              learning_rate=self.learning_rate,
                              random_seed=3)
        som.train_random(X, self.n_iter)

        # constructing the grid
        grid_min = {}
        grid_maj = {}
        for i in range(len(y)):
            tmp = som.winner(X[i])
            idx = (tmp[0], tmp[1])
            if idx not in grid_min:
                grid_min[idx] = []
            if idx not in grid_maj:
                grid_maj[idx] = []
            if y[i] == self.min_label:
                grid_min[idx].append(i)
            else:
                grid_maj[idx].append(i)

        # converting the grid to arrays
        for i in grid_min:
            grid_min[i] = np.array(grid_min[i])
        for i in grid_maj:
            grid_maj[i] = np.array(grid_maj[i])

        # filtering
        filtered = {}
        for i in grid_min:
            if i not in grid_maj:
                filtered[i] = True
            else:
                filtered[i] = (len(grid_maj[i]) + 1)/(len(grid_min[i])+1) < 1.0

        # computing densities
        densities = {}
        for i in filtered:
            if filtered[i]:
                if len(grid_min[i]) > 1:
                    paird = pairwise_distances(X[grid_min[i]])
                    densities[i] = len(grid_min[i])/np.mean(paird)**2
                else:
                    densities[i] = 10

        # all clusters can be filtered
        if len(densities) == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "all clusters filtered")
            return X.copy(), y.copy()

        # computing neighbour densities, using 4 neighborhood
        neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        pair_densities = {}
        for i in densities:
            for n in neighbors:
                j = (i[0] + n[0], i[1] + n[1]),
                if j in densities:
                    pair_densities[(i, j)] = densities[i] + densities[j]

        # computing weights
        density_keys = list(densities.keys())
        density_vals = np.array(list(densities.values()))

        # determining pair keys and density values
        pair_keys = list(pair_densities.keys())
        pair_vals = np.array(list(pair_densities.values()))

        # determining densities
        density_vals = (1.0/density_vals)/np.sum(1.0/density_vals)
        pair_dens_vals = (1.0/pair_vals)/np.sum(1.0/pair_vals)

        # computing num of samples to generate
        if len(pair_vals) > 0:
            dens_num = N_intra
            pair_num = N_inter
        else:
            dens_num = N_inter + N_intra
            pair_num = 0

        # generating the samples according to the extracted distributions
        samples = []
        while len(samples) < dens_num:
            cluster_idx = density_keys[self.random_state.choice(
                np.arange(len(density_keys)), p=density_vals)]
            cluster = grid_min[cluster_idx]
            sample_a, sample_b = self.random_state.choice(cluster, 2)
            samples.append(self.sample_between_points(
                X[sample_a], X[sample_b]))

        while len(samples) < pair_num:
            idx = pair_keys[self.random_state.choice(
                np.arange(len(pair_keys)), p=pair_dens_vals)]
            cluster_a = grid_min[idx[0]]
            cluster_b = grid_min[idx[1]]
            X_a = X[self.random_state.choice(cluster_a)]
            X_b = X[self.random_state.choice(cluster_b)]
            samples.append(self.sample_between_points(X_a, X_b))

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_grid': self.n_grid,
                'sigma': self.sigma,
                'learning_rate': self.learning_rate,
                'n_iter': self.n_iter,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
