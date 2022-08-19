"""
This module implements the SOMO method.
"""

import numpy as np

from scipy.linalg import circulant

import minisom

from ..base import pairwise_distances_mahalanobis

from ..base import coalesce, coalesce_dict, fix_density
from ..base import OverSamplingSimplex
from .._logger import logger
_logger= logger

__all__= ['SOMO']

class SOMO(OverSamplingSimplex):
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

    categories = [OverSamplingSimplex.cat_extensive,
                  OverSamplingSimplex.cat_uses_clustering,
                  OverSamplingSimplex.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 *,
                 n_grid=10,
                 sigma=0.2,
                 learning_rate=0.5,
                 n_iter=100,
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
            n_grid (int): size of grid
            sigma (float): sigma of SOM
            learning_rate (float) learning rate of SOM
            n_iter (int): number of iterations
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, 'proportion', 0)
        self.check_greater_or_equal(n_grid, 'n_grid', 2)
        self.check_greater(sigma, 'sigma', 0)
        self.check_greater(learning_rate, 'learning_rate', 0)
        self.check_greater_or_equal(n_iter, 'n_iter', 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.params = {'n_grid': n_grid,
                        'sigma': sigma,
                        'learning_rate': learning_rate,
                        'n_iter': n_iter}
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
                                  'n_grid': [5, 9, 13],
                                  'sigma': [0.4],
                                  'learning_rate': [0.3, 0.5],
                                  'n_iter': [100]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def construct_grid(self, X, y):
        """
        Construct the grid.

        Args:
            X (np.array): all feature vectors
            y (np.array): all target labels

        Returns:
            dict, dict: the minority and majority grids
        """
        # training SOM
        som = minisom.MiniSom(self.params['n_grid'],
                              self.params['n_grid'],
                              len(X[0]),
                              sigma=self.params['sigma'],
                              learning_rate=self.params['learning_rate'],
                              random_seed=3)
        som.train_random(X, self.params['n_iter'])

        # constructing the grid
        grid_min = {}
        grid_maj = {}

        for jdx, x_vec in enumerate(X):
            idx = som.winner(x_vec)
            idx = (idx[0], idx[1])
            if idx not in grid_min:
                grid_min[idx] = []
            if idx not in grid_maj:
                grid_maj[idx] = []
            if y[jdx] == self.min_label:
                grid_min[idx].append(jdx)
            else:
                grid_maj[idx].append(jdx)

        # converting the grid to arrays
        for idx in grid_min:
            grid_min[idx] = np.array(grid_min[idx])
        for idx in grid_maj:
            grid_maj[idx] = np.array(grid_maj[idx])

        return grid_min, grid_maj

    def determine_densities(self, grid_min, grid_maj, X, y):
        """
        Determine densities

        Args:
            grid_min (dict): the minority grid
            gird_maj (dict): the majority grid
            X (np.array): all feature vectors
            y (np.array): all target labels

        Returns:
            dict: the densities
        """
        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= \
                    self.metric_tensor_from_nn_params(nn_params, X, y)

        tensor = nn_params.get('metric_tensor', None)

        # filtering
        filtered = {}
        for idx, _ in grid_min.items():
            filtered[idx] = (len(grid_maj[idx]) + 1)/(len(grid_min[idx]) + 1) < 1.0

        # computing densities
        densities = {}
        for idx, filt in filtered.items():
            if filt:
                paird = pairwise_distances_mahalanobis(X[grid_min[idx]],
                                                            tensor=tensor)
                mean_value = np.mean(paird)
                if len(grid_min[idx]) > 1 and mean_value > 0:
                    densities[idx] = len(grid_min[idx])/mean_value**2
                else:
                    densities[idx] = 10

        return densities

    def determine_pair_densities(self, densities):
        """
        Determine pair densities.

        Args:
            densities (dict): the densities

        Returns:
            dict: the pair densities
        """
        # computing neighbour densities, using 4 neighborhood
        neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        pair_densities = {}
        for idx, _ in densities.items():
            for neigh in neighbors:
                jdx = (idx[0] + neigh[0], idx[1] + neigh[1])
                if jdx in densities:
                    pair_densities[(idx, jdx)] = densities[idx] + densities[jdx]

        return pair_densities

    def create_density_from_dict(self, density):
        """
        Create probability density from a dictionary.

        Args:
            density (dict): densities unnormalized in a dictionary

        Returns:
            np.array: the density values
        """
        density_vals = np.array(list(density.values()))
        density_vals = 1.0 / density_vals

        density_vals = fix_density(density_vals)

        return density_vals

    def generate_intra(self, *,
                        density_keys,
                        density_vals,
                        grid_min,
                        X,
                        n_to_sample):
        """
        Generate samples within the clusters.

        Args:
            density_keys (list): the density keys
            density_vals (np.array): the density values
            grid_min (dict): the minority grid
            X (np.array): all feature vectors
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        samples = np.zeros((0, X.shape[1]))
        clusters = self.random_state.choice(np.arange(len(density_keys)),
                                            n_to_sample,
                                            p=density_vals)

        cluster_unique, cluster_count = np.unique(clusters, return_counts=True)

        for idx, cluster_idx in enumerate(cluster_unique):
            cluster = X[grid_min[density_keys[cluster_idx]]]
            #n_dim_orig = self.n_dim
            #self.n_dim = np.min([len(cluster), n_dim_orig])
            samples_tmp = self.sample_simplex(
                                X=cluster,
                                indices=circulant(np.arange(cluster.shape[0])),
                                n_to_sample=cluster_count[idx])
            samples = np.vstack([samples, samples_tmp])
            #self.n_dim = n_dim_orig

        #while len(samples) < n_to_sample:
        #    cluster_idx = density_keys[self.random_state.choice(
        #        np.arange(len(density_keys)), p=density_vals)]
        #    cluster = grid_min[cluster_idx]
        #    sample_a, sample_b = self.random_state.choice(cluster, 2)
        #    samples.append(self.sample_between_points(
        #        X[sample_a], X[sample_b]))

        return samples

    def generate_inter_samples(self, cluster_a, cluster_b, n_to_sample):
        """
        Generate samples between two clusters.

        Args:
            cluster_a (np.array): the first cluster
            cluster_b (np.array): the second cluster
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """
        tile = np.tile(np.arange(len(cluster_b)), (len(cluster_a), 1))

        indices = (np.vstack([np.zeros(len(cluster_a)), tile.T]).T).astype(int)

        n_dim_orig = self.n_dim
        self.n_dim = 2

        samples_tmp = self.sample_simplex(
                            X=cluster_a,
                            indices=indices,
                            n_to_sample=n_to_sample,
                            X_vertices=cluster_b)

        self.n_dim = n_dim_orig

        return samples_tmp

    def generate_inter(self, *,
                        pair_keys,
                        pair_dens_vals,
                        grid_min,
                        X,
                        n_to_sample):
        """
        Generate samples between the clusters.

        Args:
            pair_keys (list): the density keys
            pair_dens_vals (np.array): the density values
            grid_min (dict): the minority grid
            X (np.array): all feature vectors
            n_to_sample (int): the number of samples to generate

        Returns:
            np.array: the generated samples
        """

        samples = np.zeros((0, X.shape[1]))

        if pair_dens_vals.shape[0] == 0:
            return samples

        clusters = self.random_state.choice(np.arange(len(pair_keys)),
                                            n_to_sample,
                                            p=pair_dens_vals)

        cluster_unique, cluster_count = np.unique(clusters, return_counts=True)

        for idx, cluster_idx in enumerate(cluster_unique):

            samples_tmp = self.generate_inter_samples(
                            cluster_a=X[grid_min[pair_keys[cluster_idx][0]]],
                            cluster_b=X[grid_min[pair_keys[cluster_idx][1]]],
                            n_to_sample=cluster_count[idx]
                            )
            samples = np.vstack([samples, samples_tmp])


        #while len(samples) < n_to_sample:
        #    idx = pair_keys[self.random_state.choice(
        #        np.arange(len(pair_keys)), p=pair_dens_vals)]
        #    cluster_a = grid_min[idx[0]]
        #    cluster_b = grid_min[idx[1]]
        #    X_a = X[self.random_state.choice(cluster_a)]
        #    X_b = X[self.random_state.choice(cluster_b)]
        #    samples.append(self.sample_between_points(X_a, X_b))

        return samples

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        grid_min, grid_maj = self.construct_grid(X, y)

        densities = self.determine_densities(grid_min, grid_maj, X, y)

        # all clusters can be filtered
        if len(densities) == 0:
            return self.return_copies(X, y, "all clusters filtered")

        pair_densities = self.determine_pair_densities(densities)

        density_vals = self.create_density_from_dict(densities)
        pair_dens_vals = self.create_density_from_dict(pair_densities)

        density_keys = list(densities.keys())
        pair_keys = list(pair_densities.keys())

        # determining num of samples to generate
        if len(pair_dens_vals) > 0:
            dens_num = int(n_to_sample / 2)
        else:
            dens_num = n_to_sample

        # generating the samples according to the extracted distributions
        samples_intra = self.generate_intra(density_keys=density_keys,
                                            density_vals=density_vals,
                                            grid_min=grid_min,
                                            X=X,
                                            n_to_sample=dens_num)

        samples_inter = self.generate_inter(pair_keys=pair_keys,
                                            pair_dens_vals=pair_dens_vals,
                                            grid_min=grid_min,
                                            X=X,
                                            n_to_sample=n_to_sample-dens_num)

        return (np.vstack([X, np.vstack([samples_intra,
                                         samples_inter])]),
                np.hstack([y, np.repeat(self.min_label,
                                        len(samples_intra) \
                                            + len(samples_inter))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_grid': self.params['n_grid'],
                'sigma': self.params['sigma'],
                'learning_rate': self.params['learning_rate'],
                'n_iter': self.params['n_iter'],
                'n_jobs': self.n_jobs,
                **OverSamplingSimplex.get_params(self)}
