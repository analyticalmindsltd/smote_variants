"""
Testing the metric tensor related functionalities.
"""

import numpy as np

import pytest

from smote_variants.base import (NearestNeighborsWithMetricTensor,
                            ClassifierImpliedDissimilarityMatrix,
                            MetricTensor,
                            MetricLearningMixin,
                            pairwise_distances_mahalanobis,
                            fix_pd_matrix,
                            reverse_matrix,
                            construct_tensor,
                            ClosestNeighborsInClasses,
                            RemoveCorrelatedColumns,
                            discrete_variable_mask,
                            estimate_mutual_information,
                            n_neighbors_func,
                            psd_mean,
                            distances_mahalanobis)

def test_distances_mahalanobis():
    """
    Test the distances_mahalanobis functions.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert distances_mahalanobis(X, X).shape[0] == 2

def test_pairwise_distances_mahalanobis():
    """
    Testing the pairwise distance calculation.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[2.0, 0.0]])

    dmatrix = pairwise_distances_mahalanobis(X, Y=X)
    assert dmatrix.shape == (2, 2)
    assert dmatrix[0, 0] == 0.0
    np.testing.assert_almost_equal(dmatrix[1, 0], np.sqrt(2))

    dmatrix = pairwise_distances_mahalanobis(X)
    assert dmatrix.shape == (2, 2)
    assert dmatrix[0, 0] == 0.0
    np.testing.assert_almost_equal(dmatrix[1, 0], np.sqrt(2))

    dmatrix = pairwise_distances_mahalanobis(X, Y=Y)
    assert dmatrix.shape == (2, 1)

    dmatrix_2 = pairwise_distances_mahalanobis(X, Y=Y, tensor=np.eye(2))
    np.testing.assert_array_equal(dmatrix, dmatrix_2)

def test_fix_pd_matrix():
    """
    Testing the fixing of positive definite matrices.
    """
    matrix = np.eye(2)
    matrix[1, 1]= -1
    matrix_fixed = fix_pd_matrix(matrix, eps=1e-4)

    np.testing.assert_almost_equal(matrix_fixed[1, 1], 1e-4)

def test_reverse_matrix():
    """
    Testing the reversion of matrices.
    """
    matrix = np.eye(2)
    matrix[1, 1]= -1
    matrix_reverted = reverse_matrix(matrix)

    assert matrix_reverted[0, 0] == -1

def test_construct_tensor():
    """
    Testing the tensor construction.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dissim_matrix = np.array([[0.0, np.sqrt(2.0), 1.0],
                                [np.sqrt(2.0), 0.0, 1.0],
                                [1.0, 1.0, 0.0]])
    tensor = construct_tensor(X, dissim_matrix)

    np.testing.assert_array_almost_equal(tensor[0], np.eye(2))

    X = np.vstack([X]*200)
    dissim_matrix = pairwise_distances_mahalanobis(X)
    tensor = construct_tensor(X, dissim_matrix)

    np.testing.assert_array_almost_equal(tensor[0], np.eye(2))

def test_classifierimplied_dissimmatrix():
    """
    Testing the classifier implied dissimilarity mixin.
    """
    cidm = ClassifierImpliedDissimilarityMatrix()
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                    [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 0, 0, 1])
    cidm.fit(X, y)

    dissim = cidm.dissimilarity_matrix(X[:3])

    np.testing.assert_almost_equal(dissim, np.array([[0.  , 0.38, 0.16],
                                            [0.38, 0.  , 0.23],
                                            [0.16, 0.23, 0.  ]]))

    assert len(cidm.get_params()) > 0

    assert len(cidm.transform_kneighbors(X)) == len(X)

def test_closest_neighbors_in_classes():
    """
    Testing the closest neighbors across classes determination.
    """
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 0, 1, 1])

    cnic = ClosestNeighborsInClasses(n_neighbors = 2)

    _, y_final = cnic.fit_transform(X, y)

    assert np.sum(y_final == 1) == 1
    assert len(cnic.get_params()) > 0

def test_remove_correlated_columns():
    """
    Testing the removal of correlated columns.
    """
    X = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    rcc = RemoveCorrelatedColumns()
    assert rcc.fit(X).transform(X).shape[1] == 1

def test_discrete_variable_mask():
    """
    Testing the detection of discrete variable masks.
    """
    X = np.array([[1.0, 1.0], [0.0, 0.0], [3.0, 0.0], [2.0, 1.0]])
    np.testing.assert_array_equal(discrete_variable_mask(X, threshold=2),
                                     np.array([False, True]))

def test_estimate_mutual_information():
    """
    Testing the mutual information estimation.
    """
    X = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 1.0],
                [0.0, 2.0], [1.0, 0.0],
                [1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    mutuali = estimate_mutual_information(X, y, n_repeats=2)

    assert mutuali[0] > mutuali[1]

    mutuali = estimate_mutual_information(X, y, n_repeats=2, normalize=False)

    assert mutuali[0] > mutuali[1]

def test_n_neighbors_func():
    """
    Testing the n_neighbors functionality.
    """
    X_base = np.array([[1.0, 0.0], [0.0, 1.0]])
    X_neighbors = np.array([[2.0, 0.0], [0.0, 2.0]])

    ind = n_neighbors_func(X_base, X_neighbors, n_neighbors=2)

    assert ind[0][0] == 0

    dist, ind = n_neighbors_func(X_base, X_neighbors,
                            n_neighbors=2, return_distance=True)

    assert dist[0][0] == 1.0

def test_psd_mean():
    """
    Testing the averaging of postivie definite matrices.
    """
    matrices = [np.array([[1.0, 0.0], [0.0, 1.0]]),
                np.array([[2.0, 0.0], [0.0, 1.0]])]

    etalon = np.array([[1.5, 0.0], [0.0, 1.0]])

    np.testing.assert_array_almost_equal(psd_mean(matrices),
                                            etalon)

def test_metrictensor():
    """
    Testing the metric tensor object.
    """
    metrict = MetricTensor()
    assert metrict.get_params()['metric_learning_method'] is None

@pytest.mark.parametrize('method', ['ITML', 'rf', 'LSML', 'cov',
                                    'cov_min', 'MI_weighted', 'id',
                                    'ITML_mi', 'NCA', 'gmean'])
def test_metrictensor_learning(method):
    """
    Testing the metric learning

    Args:
        method (str): metric learning algorithm
    """
    X = np.array([[0.0, 1.0], [1.5, 2.0], [0.5, 1.0],
                [0.0, 2.0], [1.0, 0.0],
                [1.0, 2.0], [0.0, 1.5], [1.5, 0.0]])
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    metrict = MetricTensor(metric_learning_method=method)
    assert metrict.tensor(X, y).shape == (2, 2)

def test_metriclearningmixin():
    """
    Testing the metric learning mixin.
    """
    X = np.array([[0.0, 1.0], [1.5, 2.0], [0.5, 1.0],
                [0.0, 2.0], [1.0, 0.0],
                [1.0, 2.0], [0.0, 1.5], [1.5, 0.0]])
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])

    mlm = MetricLearningMixin()

    nn_params = {'metric_learning_method': 'cov',
                    'metric': 'precomputed'}

    tensor = mlm.metric_tensor_from_nn_params(nn_params, X, y)
    assert tensor.shape == (2, 2)

    nn_params = {'metric_tensor': np.eye(2)}
    tensor = mlm.metric_tensor_from_nn_params(nn_params, X, y)
    assert tensor.shape == (2, 2)

    nn_params = {'metric': 'minkowski',
                'metric_tensor': None}
    assert mlm.metric_tensor_from_nn_params(nn_params, X, y) is None

    assert len(mlm.get_params()) == 0

def test_nn_metric_tensor():
    """
    Testing the nearest neighbors with metric tensor.
    """
    X_base = np.array([[1.0, 0.0], [0.0, 1.0]])
    X_neighbors = np.array([[2.0, 0.0], [0.0, 2.0]])

    nearestn = NearestNeighborsWithMetricTensor(n_neighbors=2)

    nearestn.fit(X_neighbors)

    ind = nearestn.kneighbors(X_base, return_distance=False)

    assert ind[0][0] == 0

    dist, ind = nearestn.kneighbors(X_base)

    assert dist[0][0] == 1.0

    ind = nearestn.radius_neighbors(X_base, radius=1.2,
                                        return_distance=False)

    assert len(ind[0]) == 1

    dist, ind = nearestn.radius_neighbors(X_base, radius=1.2)

    assert dist[0][0] == 1.0

    nearestn = NearestNeighborsWithMetricTensor(n_neighbors=2,
                                                metric='precomputed',
                                                metric_tensor=np.eye(2))

    nearestn.fit(X_neighbors)

    ind = nearestn.kneighbors(X_base, return_distance=False)

    assert ind[0][0] == 0

    dist, ind = nearestn.kneighbors(X_base)

    assert dist[0][0] == 1.0

    ind = nearestn.radius_neighbors(X_base, radius=1.2,
                                        return_distance=False)

    assert len(ind[0]) == 1

    dist, ind = nearestn.radius_neighbors(X_base, radius=1.2)

    assert dist[0][0] == 1.0

    assert len(nearestn.get_params()) > 0
