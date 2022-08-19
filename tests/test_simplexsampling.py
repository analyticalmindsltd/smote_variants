"""
Testing the simplex sampling functionalities.
"""

import numpy as np

import pytest

from smote_variants.base import (array_array_index,
                            base_idx_neighbor_idx_simplices,
                            all_neighbor_simplices_real_idx,
                            reweight_simplex_vertices,
                            cartesian_product, vector_choice,
                            simplex_volume, simplex_volumes,
                            SimplexSamplingMixin,
                            random_samples_from_simplices,
                            add_samples)

def test_array_array_index():
    """
    Testing the array indexing.
    """
    array = np.array([[0, 1], [2, 3]])
    indices = np.array([[0, 1], [0, 1]])

    np.testing.assert_array_equal(array,
                                    array_array_index(array, indices))

def test_base_idx_neighbor_idx_simplices():
    """
    Testing the simplex generation.
    """
    all_simplices = base_idx_neighbor_idx_simplices(n_base=5,
                                                    n_neighbors=2,
                                                    n_dim=2)
    assert len(all_simplices) == 5

def test_all_neighbors_real_idx():
    """
    Testing the simplex generation in vector index coordinates.
    """
    indices = np.array([[0, 10, 11],
                        [1, 10, 11],
                        [2, 10, 11],
                        [3, 10, 11],
                        [4, 10, 11]])
    real_indices = all_neighbor_simplices_real_idx(n_dim=2,
                                                    indices=indices)

    np.testing.assert_array_equal(real_indices[0], np.array([0, 10]))
    np.testing.assert_array_equal(real_indices[1], np.array([0, 11]))

def test_simplex_volume():
    """
    Testing the volume calculation.
    """
    simplex = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert simplex_volume(simplex) == np.sqrt(2.0)

def test_simplex_volumes():
    """
    Testing the volume calculation for multiple simplices.
    """
    simplices = np.array([np.array([[0.0, 1.0], [1.0, 0.0]])])
    assert simplex_volumes(simplices)[0] == np.sqrt(2.0)

    simplices3 = np.array([np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])])
    assert simplex_volumes(simplices3)[0] == 0.5

    assert len(simplex_volumes(np.array([]))) == 0

def test_reweight_simplex_vertices():
    """
    Testing the reweighting of vertices.
    """
    base_vectors = np.array([[0.0, 0.0],
                            [0.0, 1.0]])
    simplices = np.array([[0, 1]])
    vertex_weights = np.array([1.0, 0.5])

    results = reweight_simplex_vertices(base_vectors,
                                        simplices,
                                        None,
                                        vertex_weights)

    etalon = np.array([[[0.0, 0.0],
                        [0.0, 0.5]]])

    np.testing.assert_array_equal(results, etalon)

def test_cartesian_product():
    """
    Testing the Cartesian product calculation.
    """
    arr0 = np.array([0, 1])
    arr1 = np.array([3])

    cartesian = cartesian_product(arr0, arr1)

    etalon = np.array([[0, 3], [1, 3]])

    np.testing.assert_array_equal(cartesian, etalon)

def test_vector_choice():
    """
    Testing the vector choice functionality.
    """
    values = np.array([0, 1, 2, 3])
    dist = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    result = vector_choice(values, dist)

    np.testing.assert_array_equal(result, np.array([0, 3]))

# note the spacial nature of this problem, the vectors [3] and [4]
# are neighbors of each other multiple times, which leads to a different
# proportion of samples when generated with the deterministic and
# random within simplex sampling
X_SSM = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0],
                    [0.01, 2.00], [0.01, 2.01]])
INDICES_SSM = np.array([[0, 2, 1], [1, 0, 2], [2, 0, 1], [3, 4, 4], [4, 3, 3]])
SMALL_VOLUME_EDGE_WEIGHT_DET = 0.4
SMALL_VOLUME_EDGE_WEIGHT_RAN = 0.25
N_TO_SAMPLE_SSM = 1000

def test_simplexsamplingmixin_dim2_uniform_deterministic():
    """
    Testing uniform and deterministic sampling in 2D.
    """
    ssm = SimplexSamplingMixin(simplex_sampling='uniform',
                                within_simplex_sampling='deterministic',
                                n_dim=2,
                                random_state=5)

    assert len(ssm.get_params()) > 0

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM)

    assert len(samples) == N_TO_SAMPLE_SSM

    splitting_point = samples[samples[:, 0] == 0.0][0][1]

    # the possible splitting points for deterministic setup
    possible_splits = []
    for idx in range(2, N_TO_SAMPLE_SSM):
        step = 1.0/idx
        splits = [1.0 + step*count for count in range(1, idx)]
        possible_splits.extend(splits)

    assert splitting_point in possible_splits

    # estimating the reasonable number of samples from an edge
    frac_small_samples = np.sum(samples[:, 0] == 0.01)/N_TO_SAMPLE_SSM
    frac_diff_perc = abs(SMALL_VOLUME_EDGE_WEIGHT_DET - frac_small_samples)
    frac_diff_perc = frac_diff_perc / SMALL_VOLUME_EDGE_WEIGHT_DET

    assert frac_diff_perc < 0.05

def test_simplexsamplingmixin_dim2_uniform_random():
    """
    Testing uniform and random sampling in 2D.
    """
    ssm = SimplexSamplingMixin(simplex_sampling='uniform',
                                within_simplex_sampling='random',
                                n_dim=2,
                                random_state=5)

    assert len(ssm.get_params()) > 0

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM)

    assert len(samples) == N_TO_SAMPLE_SSM

    splitting_points = samples[samples[:, 0] == 0.0][:, 1]
    assert np.all(np.logical_and(splitting_points >= 1,
                                    splitting_points <= 2))


    # estimating the reasonable number of samples from an edge
    frac_small_samples = np.sum(samples[:, 0] == 0.01)/N_TO_SAMPLE_SSM
    frac_diff_perc = abs(SMALL_VOLUME_EDGE_WEIGHT_RAN - frac_small_samples)
    frac_diff_perc = frac_diff_perc / SMALL_VOLUME_EDGE_WEIGHT_RAN

    assert frac_diff_perc < 0.05

def test_simplexsamplingmixin_dim2_volume_deterministic():
    """
    Testing volume weighted and deterministic sampling in 2D.
    """
    ssm = SimplexSamplingMixin(simplex_sampling='volume',
                                within_simplex_sampling='deterministic',
                                n_dim=2,
                                random_state=5)

    assert len(ssm.get_params()) > 0

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM)

    assert len(samples) == N_TO_SAMPLE_SSM

    splitting_point = samples[samples[:, 0] == 0.0][0][1]

    # the possible splitting points for deterministic setup
    possible_splits = []
    for idx in range(2, N_TO_SAMPLE_SSM):
        step = 1.0/idx
        splits = [1.0 + step*count for count in range(1, idx)]
        possible_splits.extend(splits)

    assert splitting_point in possible_splits

    # estimating the reasonable number of samples from an edge
    frac_small_volume_samples = np.sum(samples[:, 0] == 0.01)/N_TO_SAMPLE_SSM

    assert frac_small_volume_samples < 0.05

def test_simplexsamplingmixin_dim2_volume_random():
    """
    Testing volume weighted and random sampling in 2D.
    """
    ssm = SimplexSamplingMixin(simplex_sampling='volume',
                                within_simplex_sampling='random',
                                n_dim=2,
                                random_state=5)

    assert len(ssm.get_params()) > 0

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM)

    assert len(samples) == N_TO_SAMPLE_SSM

    splitting_points = samples[samples[:, 0] == 0.0][:, 1]
    assert np.all(np.logical_and(splitting_points >= 1,
                                    splitting_points <= 2))


    # estimating the reasonable number of samples from an edge
    frac_small_volume_samples = np.sum(samples[:, 0] == 0.01)/N_TO_SAMPLE_SSM

    assert frac_small_volume_samples < 0.05

def test_simplexsamplingmixin_nonsense():
    """
    Testing nonsense parameters.
    """
    with pytest.raises(ValueError):
        ssm = SimplexSamplingMixin(simplex_sampling=None,
                                within_simplex_sampling='random',
                                n_dim=2,
                                random_state=5)

        ssm.sample_simplex(X=X_SSM,
                            indices=INDICES_SSM,
                            n_to_sample=N_TO_SAMPLE_SSM)

    with pytest.raises(ValueError):
        ssm = SimplexSamplingMixin(simplex_sampling='volume',
                                within_simplex_sampling=None,
                                n_dim=2,
                                random_state=5)

        ssm.sample_simplex(X=X_SSM,
                            indices=INDICES_SSM,
                            n_to_sample=N_TO_SAMPLE_SSM)

    with pytest.raises(ValueError):
        ssm = SimplexSamplingMixin(simplex_sampling='volume',
                                within_simplex_sampling='deterministic',
                                n_dim=3,
                                random_state=5)

        ssm.sample_simplex(X=np.random.random(size=(10, 2)),
                            indices=np.random.randint(5, size=(10, 5)),
                            n_to_sample=N_TO_SAMPLE_SSM)

def test_simplexsamplingmixin_dim2_base_weights():
    """
    Testing the use of base weights.
    """
    ssm = SimplexSamplingMixin(simplex_sampling='volume',
                                within_simplex_sampling='deterministic',
                                n_dim=2,
                                random_state=5)

    assert len(ssm.get_params()) > 0

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM,
                                    base_weights=np.array([0, 0, 0,
                                                        0.5, 0.5]))

    assert len(samples) == N_TO_SAMPLE_SSM
    assert np.all(samples[:, 0] == 0.01)

def test_simplexsamplingmixin_dim2_vertex_weights():
    """
    Testing the use of vertex weights.
    """
    ssm = SimplexSamplingMixin(simplex_sampling='volume',
                                within_simplex_sampling='random',
                                n_dim=2,
                                random_state=5)

    assert len(ssm.get_params()) > 0

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM,
                                    base_weights=np.array([0, 0, 0, 0.5, 0.5]),
                                    vertex_weights=np.array([1.0, 1.0, 1.0,
                                                            1.0, 0.5]))

    assert len(samples) == N_TO_SAMPLE_SSM
    samples = samples[samples[:, 0] == 0.01]
    n_above = np.sum(samples[:, 1] >= 2.005)
    n_below = np.sum(samples[:, 1] <= 2.005)

    assert abs(n_below /n_above - 3)/3 < 0.05

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM,
                                    base_weights=np.array([0, 0, 0, 1.0, 0]),
                                    vertex_weights=np.array([1.0, 1.0, 1.0,
                                                            1.0, 0.5]),
                                    X_vertices=X_SSM.copy())

    assert len(samples) == N_TO_SAMPLE_SSM
    assert np.all(np.logical_and(samples[:, 1] <= 2.005,
                                    samples[:, 1] >= 2.0))

    ssm = SimplexSamplingMixin(simplex_sampling='uniform',
                                within_simplex_sampling='random',
                                n_dim=2,
                                random_state=5)

    samples = ssm.sample_simplex(X=X_SSM,
                                    indices=INDICES_SSM,
                                    n_to_sample=N_TO_SAMPLE_SSM,
                                    vertex_weights=np.array([1.0, 1.0, 1.0,
                                                            1.0, 0.5]),
                                    X_vertices=X_SSM.copy())

    assert len(samples) == N_TO_SAMPLE_SSM

    samples = samples[samples[:, 0] == 0.01]
    n_above = np.sum(samples[:, 1] >= 2.005)
    n_below = np.sum(samples[:, 1] <= 2.005)

    assert abs(n_below /n_above - 3)/3 < 0.05

def test_random_samples_from_simplices():
    """
    Testing sampling within a triangle and checking if the samples
    fall on the correct side (or on) the hypotenuse.
    """
    X = np.array([[0, 1], [1, 0], [0, 0]])
    X_vertices = np.array([[0, 1], [1, 0], [1, 1]])
    simplices = np.array([[0, 1, 2]]*10)
    vertex_weights_0 = np.array([1.0, 1.0, 0.0])

    sample = random_samples_from_simplices(X, simplices)
    assert np.all(np.dot(sample, np.array([1.0, 1.0])) - 1 < 0)

    sample = random_samples_from_simplices(X, simplices, X_vertices=X)
    assert np.all(np.dot(sample, np.array([1.0, 1.0])) - 1 < 0)

    sample = random_samples_from_simplices(X, simplices, X_vertices=X_vertices)
    assert np.all(np.dot(sample, np.array([1.0, 1.0])) - 1 > 0)

    sample = random_samples_from_simplices(X, simplices,
                                            vertex_weights=vertex_weights_0)
    np.testing.assert_almost_equal(np.dot(sample, np.array([1.0, 1.0])) - 1, 0)

    sample = random_samples_from_simplices(X, simplices,
                                            X_vertices = X_vertices,
                                            vertex_weights=vertex_weights_0)
    np.testing.assert_almost_equal(np.dot(sample, np.array([1.0, 1.0])) - 1, 0)

def test_add_samples():
    """
    Testing the add samples functionality (deterministic sample generation)
    """
    pairs = np.array([(0, 1), (0, 2), (1, 2)])
    counts = np.array([1, 1, 2])
    count = 2
    X = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    X_vertices = np.array([[0.0, 2.0], [2.0, 0.0], [2.0, 2.0]])
    vertex_weights = np.array([1.0, 1.0, 0.5])

    samples = add_samples(pairs=pairs,
                            counts=counts,
                            count=count,
                            X=X,
                            X_vertices=X)

    np.testing.assert_array_equal(samples, np.array([[1.0, 1.0/3.0],
                                                    [1.0, 2.0/3.0]]))

    samples = add_samples(pairs=pairs,
                            counts=counts,
                            count=count,
                            X=X,
                            X_vertices=X_vertices)

    np.testing.assert_array_equal(samples, np.array([[1 + 1.0/3.0, 2*1.0/3.0],
                                                    [1 + 2.0/3.0, 2*2.0/3.0]]))

    samples = add_samples(pairs=pairs,
                            counts=counts,
                            count=count,
                            X=X,
                            X_vertices=X,
                            vertex_weights=vertex_weights)

    np.testing.assert_array_equal(samples, np.array([[1.0, 0.5*1.0/3.0],
                                                    [1.0, 0.5*2.0/3.0]]))

    samples = add_samples(pairs=pairs,
                            counts=counts,
                            count=count,
                            X=X,
                            X_vertices=X_vertices,
                            vertex_weights=vertex_weights)

    np.testing.assert_array_equal(samples, np.array([[1 + 0.5*1.0/3.0, 0.5*2*1.0/3.0],
                                                    [1 + 0.5*2.0/3.0, 0.5*2*2.0/3.0]]))

def test_add_gaussian():
    """
    Test the add gaussian function.
    """
    samples = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    simplex_sampling = SimplexSamplingMixin()

    np.testing.assert_array_equal(samples,
                                  simplex_sampling.add_gaussian_noise(samples))
