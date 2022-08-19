
"""
This module implements all simplex sampling related functionalities.
"""

import itertools

import numpy as np

from ._base import RandomStateMixin, coalesce
from .._logger import logger as _logger

__all__= ['array_array_index',
          'base_idx_neighbor_idx_simplices',
          'all_neighbor_simplices_real_idx',
          'reweight_simplex_vertices',
          'cartesian_product',
          'vector_choice',
          'simplex_volume',
          'simplex_volumes',
          'SimplexSamplingMixin',
          'random_samples_from_simplices',
          'add_samples']

def array_array_index(array, indices):
    """
    Take samples from an array by indices row by row

    Args:
        array (np.array): array to sample from
        indices (np.array): array of indices

    Returns:
        array (indices.shape): the sampled values
    """
    stride = np.arange(indices.shape[0])*array.shape[1]
    indices_mod = indices + stride[:, None]
    indices_flat = indices_mod.flatten()
    return array.flatten()[indices_flat].reshape(indices.shape)

def base_idx_neighbor_idx_simplices(n_base, n_neighbors=5, n_dim=2):
    """
    Returns all possible simplices of a given dimensionality

    Args:
        n_base (int): number of base records
        n_neighbors (int): number of neighbors
        n_dim (int): simplex dimensionality (for triangles 3)

    Returns:
        np.array: first column: base record index, the rest are neighbor
                                indices
    """
    combinations = np.array(list(itertools.combinations(np.arange(1,
                                                        n_neighbors),
                                                        n_dim-1))).astype(int)
    base_indices = np.repeat(np.arange(n_base), len(combinations))
    all_simplices = np.vstack([base_indices,
                                np.tile(combinations, (n_base, 1)).T]).T
    return all_simplices

def all_neighbor_simplices_real_idx(n_dim, indices):
    """
    Determine the real indices of all simplices

    Args:
        n_dim (int): simplex order
        indices (np.array): neighbor indices

    Returns:
        np.array: the array of simplex indices of the real vectors
    """
    all_simplices = base_idx_neighbor_idx_simplices(n_base=indices.shape[0],
                                                n_neighbors=indices.shape[1],
                                                n_dim=n_dim)
    base_vector_indices = all_simplices[:, 0]
    neighbors_indices = indices[base_vector_indices]
    neighbors_indices = array_array_index(neighbors_indices,
                                            all_simplices[:,1:])
    simplices_real_indices = np.vstack([base_vector_indices.T,
                                        neighbors_indices.T]).T

    return simplices_real_indices

def simplex_volume(simplex):
    """
    Computes the volume of a simplex

    Args:
        simplex (np.array): vertices of the simplex

    Returns:
        float: the volume of the simplex
    """
    simplex_mod = simplex[:-1] - simplex[-1]
    gram = np.dot(simplex_mod, simplex_mod.T)
    det = np.linalg.det(gram)
    return np.sqrt(det)/np.math.factorial(simplex.shape[0]-1)

def simplex_volumes(simplices):
    """
    Computes the volume of an array of simplices

    Args:
        simplices (np.array): array of the simplices' vertices

    Returns:
        np.array: the volumes of the simplices
    """
    if simplices.shape[0] == 0:
        return np.array([], dtype=float)
    if simplices[0].shape[0] == 2:
        return np.sqrt(np.sum((simplices[:,0,:] - simplices[:,1,:])**2,
                                axis=1))
    return np.array([simplex_volume(x) for x in simplices])

def reweight_simplex_vertices(X,
                                simplices,
                                X_vertices=None,
                                vertex_weights=None):
    """
    Simplex vertices are split into two sets, and the weights of the ones in
    the set X_vertices vary. The weight 1 means full contribution, a weight
    0 < w < 1 a proportionally smaller contribution.

    Args:
        X (np.array): the base vectors
        simplices (np.array): the vertices of the simplices
        X_vertices (np.array): the neighbor vectors
        vertex_weights (np.array): the weights of the neighbor vectors

    Returns:
        np.array: the simplices with scaled neighbor edges
    """
    if X_vertices is None:
        X_vertices = X

    # extracting the weights of each neighbor vector
    vertex_weights_simplices = vertex_weights[simplices[:,1:]]

    X_base = X[simplices[:,0]]
    X_neighbors = X_vertices[simplices[:,1:]]

    # calculating the direction vectors (neighbor vectors - base vector)
    diff_vectors = X_neighbors - X_base[:,None]

    # reweighting the direction vectors of neighbors
    weighted = (diff_vectors.T*vertex_weights_simplices.T).T

    # shifint the scaled neighbors back by the base vectors
    reverted = weighted + X_base[:,None]

    # concatenating the base vectors and the scaled vectors
    return np.concatenate([X_base[:,None], reverted], axis=1)

def cartesian_product(*arrays):
    """
    Computes the Cartesian product of arrays

    Args:
        arrays: variable list of arrays

    Returns:
        array(np.array): the array of Cartesian products
    """
    length = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [length], dtype=dtype)
    for idx, array in enumerate(np.ix_(*arrays)):
        arr[...,idx] = array
    return arr.reshape(-1, length)

def vector_choice(arr,
                    p, # pylint: disable=invalid-name
                    random_state=None):
    """
    Vector choice

    Args:
        arr (np.array): the array to choose from
        p (np.array): the array of probabilities

    Returns:
        np.array: the choosen elements
    """
    if random_state is None:
        random_state = np.random.RandomState(5)

    prob_cum = np.cumsum(p, axis=1)
    rands = random_state.rand(p.shape[0])
    return arr[np.argmax((rands < prob_cum.T).T, axis=1)]

def random_samples_from_simplices(X,
                                    simplices,
                                    X_vertices=None,
                                    vertex_weights=None,
                                    random_state=None):
    """
    Generate random samples within the simplices.

    Args:
        X (np.array): the base vectors
        simplices (np.array): the indices of the simplex nodes
        X_vertices (np.array): possibly different array of neighboring
                                vertices
        vertex_weights (np.array): weights of the neighboring vertices
        random_state (np.random.RandomState/None): the random state to
                                                    use

    Returns:
        np.array: the random samples
    """
    if random_state is None:
        random_state = np.random.RandomState(5)

    weights = random_state.random_sample(simplices.shape)
    weights = weights/np.sum(weights, axis=1)[:, None]

    if X_vertices is None and vertex_weights is None:
        X_simp = X[simplices]
    elif X_vertices is None and vertex_weights is not None:
        X_simp = reweight_simplex_vertices(X,
                                            simplices,
                                            X,
                                            vertex_weights)
    elif X_vertices is not None and vertex_weights is None:
        n_records = X.shape[0]
        X_all = np.vstack([X, X_vertices])
        simplices[:,1:] = simplices[:,1:] + n_records
        X_simp = X_all[simplices]
        simplices[:,1:] = simplices[:,1:] - n_records
    else:
        X_simp = reweight_simplex_vertices(X,
                                        simplices,
                                        X_vertices,
                                        vertex_weights)

    weighted_simplices = X_simp*weights[:, :, None]

    samples = np.sum(weighted_simplices, axis=1)

    return samples

def add_samples(*,
                pairs,
                counts,
                count,
                X,
                X_vertices,
                vertex_weights=None):
    """
    Create samples for 'count' splitting points
    between the edge pairs listed in 'pairs'

    Args:
        pairs (np.array(tuple)): unique simplices
        counts (np.array(int)): cardinalities
        count (int): one particular count
        X (np.array): base points
        X_vertices (np.array): possibly different neighbors
        vertex_weights (np.array): weights of the vertices

    Returns:
        np.array: the generated samples
    """
    pairs_filtered = pairs[counts == count]

    splits = 1.0/(count + 1)*(np.arange(1, count+1))

    base_points = X[pairs_filtered[:,0]]
    neighbor_points = X_vertices[pairs_filtered[:,1]]

    if vertex_weights is None:
        weights = np.repeat(1.0, len(pairs_filtered))
    else:
        weights = vertex_weights[pairs_filtered[:,1]]

    diffs = neighbor_points - base_points
    diffweight = (diffs.T*weights).T

    samples_by_count = [base_points + diffweight*s for s in splits]
    return np.vstack(samples_by_count)

class SimplexSamplingMixin(RandomStateMixin):
    """
    The mixin class for all simplex sampling based techniques.
    """
    def __init__(self,
                *,
                n_dim=2,
                simplex_sampling='uniform',
                within_simplex_sampling='deterministic',
                gaussian_component=None,
                random_state=None):
        """
        Base mixin of the methods implementing simplex sampling.

        Args:
            n_dim (int): the dimensions of the simplices
                            (2 for line segments)
            simplex_sampling (str): simplex sampling method
                                    ('uniform'/'volume'/None)
            within_simplex_sampling (str): within simplex sampling
                                    method ('random'/'deterministic')
        """
        RandomStateMixin.__init__(self, random_state)
        self.n_dim = n_dim
        self.simplex_sampling = simplex_sampling
        self.within_simplex_sampling = within_simplex_sampling
        self.gaussian_component = coalesce(gaussian_component, {})

    def get_params(self, deep=False):
        """
        Returns the params of the object.

        Args:
            deep (bool): deep discovery of params

        Returns:
            dict: the parameter dict
        """
        sampling_params = {'n_dim': self.n_dim,
                    'simplex_sampling': self.simplex_sampling,
                    'within_simplex_sampling': self.within_simplex_sampling,
                    'gaussian_component': self.gaussian_component}
        return {'ss_params': sampling_params,
                **RandomStateMixin.get_params(self, deep)}

    def determine_simplex_distribution(self, X, simplices):
        """
        Determine the simplex distribution.

        Args:
            X (np.array): the base vectors
            simplices (np.array): the node indices of the simplices

        Returns:
            np.array: the distribution for sampling the simplices
        """
        if self.simplex_sampling == 'uniform':
            return np.repeat(1.0/len(simplices), len(simplices))
        if self.simplex_sampling == 'volume':
            return simplex_volumes(X[simplices])
        raise ValueError(f"simplex sampling with weighting"\
                            f"{self.simplex_sampling} not implemented yet")

    def all_simplices_node_weights(self, indices, simplex_weights, n_dim):
        """
        Determine all simplices and the node weights.

        Args:
            indices (np.array): the neighborhood structure
            simplex_weights (np.array): the simplex node weights
            n_dim (int): the dimensionality of the simplices

        Returns:
            np.array, np.array: all simplices and the node weights
        """
        all_simplices = all_neighbor_simplices_real_idx(n_dim, indices)
        if simplex_weights is not None:
            all_simplices_nidx = base_idx_neighbor_idx_simplices(n_base=indices.shape[0],
                                                            n_neighbors=indices.shape[1],
                                                            n_dim=n_dim)
            base_indices = all_simplices_nidx[:,0]
            neighbor_indices = all_simplices_nidx[:,1:]
            node_weights = array_array_index(simplex_weights[base_indices], neighbor_indices)
            node_weights = np.prod(node_weights, axis=1)
        else:
            node_weights = np.repeat(1.0, all_simplices.shape[0])

        return all_simplices, node_weights

    def determine_weights(self, *, X, base_weights, X_vertices, all_simplices, indices):
        """
        Determine the weights.

        Args:
            X (np.array): the base vectors
            base_weights (np.array): the base weights
            X_vertices (np.array): the vertices
            all_simplices (np.array): all simplices
            indices (np.array): the neighborhood structure

        Returns:
            np.array: the weights
        """
        if base_weights is None:
            # base_weights overwrites the simplex_sampling parameter
            if X_vertices is None:
                weights = self.determine_simplex_distribution(X, all_simplices)
            else:
                # joining all vectors to make things easier
                simplices_joint = all_simplices.copy()
                simplices_joint[:, 1:] = simplices_joint[:, 1:]\
                                        + indices.shape[0]
                X_joint = np.vstack([X, X_vertices])
                weights = self.determine_simplex_distribution(X_joint,
                                                            simplices_joint)
        else:
            n_records = len(base_weights)
            n_simplices = len(all_simplices)
            weights = np.repeat(base_weights, int(n_simplices/n_records))

        return weights

    def simplices(self,
                    X,
                    n_to_sample,
                    *,
                    base_weights=None,
                    indices=None,
                    X_vertices=None,
                    simplex_weights=None,
                    n_dim=None):
        """
        Sampling the simplices.

        Args:
            X (np.array): the base vectors
            n_to_sample (int): the number of samples to generate
            base_weights (np.array): the weights of the base vectors
                        (overwrites the simplex_sampling parameter)
            indices (np.array): the neighborhood relations
            X_vertices (np.array): possibly different array for neighbor
                                    vectors
            simplex_weights (np.array): weights of the simplices
        """
        all_simplices, node_weights = \
            self.all_simplices_node_weights(indices, simplex_weights, n_dim=n_dim)

        weights = self.determine_weights(X=X,
                                         base_weights=base_weights,
                                         X_vertices=X_vertices,
                                         all_simplices=all_simplices,
                                         indices=indices)

        weights = weights * node_weights

        # sample the simplices
        choices = np.arange(all_simplices.shape[0])
        selected_indices = self.random_state.choice(choices,
                                                    n_to_sample,
                                                    p=weights/np.sum(weights))
        return all_simplices[selected_indices]

    def add_gaussian_noise(self, samples):
        """
        Add Gaussian noise according to the specification.

        Args:
            samples (np.array): the generated samples

        Returns:
            np.array: the noisy samples
        """

        if 'sigma' in self.gaussian_component:
            sigma = self.gaussian_component['sigma']
            return samples + self.random_state.normal(size=samples.shape) * sigma
        if 'sigmas' in self.gaussian_component:
            sigmas = self.gaussian_component['sigmas']
            return samples + self.random_state.normal(size=samples.shape) * sigmas

        return samples

    def deterministic_samples_dim_2(self,
                                    X,
                                    simplices,
                                    X_vertices=None,
                                    vertex_weights=None):
        """
        Carry out uniform sampling on line segments.

        Args:
            X (np.array): the base vectors
            simplices (np.array): the indexes of the simplex nodes
            X_vertices (np.array): array of neighbor vectors
            vertex_weights (np.array): weights of the neighbor vectors

        Returns:
            np.array: the random samples
        """
        samples = []

        base_indices = simplices[:, 0]
        neighbor_indices = simplices[:, 1]

        pairs = np.array(list(zip(base_indices, neighbor_indices)))

        if X_vertices is None or X is X_vertices:
            pairs.sort()
            X_vertices = X

        values, counts = np.unique(pairs, axis=0, return_counts=True)

        max_count = np.max(counts)

        samples = [add_samples(pairs=values,
                                counts=counts,
                                count=count,
                                X=X,
                                X_vertices=X_vertices,
                                vertex_weights=vertex_weights) for count in range(1, max_count+1)\
                                                         if count in counts]
        return np.vstack(samples)

    def deterministic_samples_dim_1(self,
                                    X,
                                    simplices):
        """
        Deterministic samples for 1-st order simplices.

        Args:
            X (np.array): base vectors
            simplices (np.array): the simplices

        Returns:
            np.array: the sampled simlices
        """
        return X[simplices[:, 0]]

    def deterministic_samples_from_simplices(self,
                                                *,
                                                X,
                                                simplices,
                                                X_vertices=None,
                                                vertex_weights=None,
                                                n_dim=None):
        """
        Uniform sampling on simplices.

        Args:
            X (np.array): the base vectors
            simplices (np.array): the indexes of the simplex nodes
            X_vertices (np.array): array of neighbor vectors
            vertex_weights (np.array): weights of the neighbor vectors

        Returns:
            np.array: the random samples
        """
        if n_dim == 2:
            samples= self.deterministic_samples_dim_2(X,
                                                    simplices,
                                                    X_vertices,
                                                    vertex_weights)
            return samples

        if n_dim == 1:
            return self.deterministic_samples_dim_1(X, simplices)

        raise ValueError(f"deterministic calculations for dim {n_dim} are\
                            not implemented yet")

    def sample_simplex(self,
                X,
                *,
                indices,
                n_to_sample,
                base_weights=None,
                vertex_weights=None,
                X_vertices=None,
                simplex_weights=None):
        """
        Carry out the simplex sampling.

        Args:
            X (np.array): the base vectors
            n_to_sample (int): the number of samples to generate
            base_weights (np.array): the weights of the base vectors
            vertex_weights (np.array): weights of the neighbor vectors
            X_vertices (np.array): array of neighbor vectors
            simplex_weights (np.array): weights of the simplices

        Returns:
            np.array: the generated samples
        """

        if n_to_sample == 0:
            return np.zeros(shape=(0, X.shape[1]))

        n_dim = self.n_dim
        n_dim = np.min([n_dim, indices.shape[1]])

        if n_dim != self.n_dim:
            _logger.info("%s: The simplex order was updated from %d to %d",
                        self.__class__.__name__, self.n_dim, n_dim)

        _logger.info("%s: simplex sampling with n_dim %d",
                        self.__class__.__name__, n_dim)

        simplices = self.simplices(X,
                                n_to_sample,
                                base_weights=base_weights,
                                indices=indices,
                                X_vertices=X_vertices,
                                simplex_weights=simplex_weights,
                                n_dim=n_dim)

        if self.within_simplex_sampling == 'random':
            samples = random_samples_from_simplices(X,
                                                simplices,
                                                X_vertices,
                                                vertex_weights,
                                                self.random_state)
        elif self.within_simplex_sampling == 'deterministic':
            samples = self.deterministic_samples_from_simplices(X=X,
                                                                simplices=simplices,
                                                                X_vertices=X_vertices,
                                                                vertex_weights=vertex_weights,
                                                                n_dim=n_dim)
        else:
            msg = "Within simplex sampling strategy"\
                f" {self.within_simplex_sampling} is not implemented yet."
            raise ValueError(msg)
        return self.add_gaussian_noise(samples)
