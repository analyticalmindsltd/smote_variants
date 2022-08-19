"""
This module contains all experimental and stable functionalities related
to metric learning.
"""
import warnings

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression

from scipy.stats import rankdata

from metric_learn import ITML_Supervised, LSML_Supervised, NCA

from ._base import instantiate_obj, coalesce
from .._logger import logger
_logger = logger

__all__= ['NearestNeighborsWithMetricTensor',
          'ClassifierImpliedDissimilarityMatrix',
          'MetricTensor',
          'MetricLearningMixin',
          'pairwise_distances_mahalanobis',
          'distances_mahalanobis',
          'fix_pd_matrix',
          'reverse_matrix',
          'construct_tensor',
          'ClosestNeighborsInClasses',
          'RemoveCorrelatedColumns',
          'discrete_variable_mask',
          'estimate_mutual_information',
          'n_neighbors_func',
          'psd_mean']

def pairwise_distances_mahalanobis(X, *, Y=None, tensor=None):
    """
    Returns the pairwise distances of two arrays of vectors using the
    provided metric tensor matrix. If no Y paramter is provided, the
    distance matrix of the paramter X is returned.

    Args:
        X (np.array): array of vectors to calculate distances for
        Y (np.array/None): optional second array of vectors
        tensor (np.array): the metric tensor to be used

    Returns:
        np.array: the distance matrix
    """
    if Y is None:
        Y = X
    if tensor is None:
        tensor = np.eye(X.shape[1])
    tmp= (X[:,None] - Y)
    return np.sqrt(np.einsum('ijk,ijk -> ij', tmp, np.dot(tmp, tensor)))

def distances_mahalanobis(X, Y, tensor=None):
    """
    Returns the distances of the corresponding vectors.

    Args:
        X (np.array): array of vectors
        Y (np.array): another array of vectors
        tensor (np.array): the metric tensor to be used

    Returns:
        np.array: the distances
    """
    if tensor is None:
        tensor = np.eye(X.shape[1])
    tmp = X - Y
    return np.sqrt(np.einsum('ij,jk,ik -> i', tmp, tensor, tmp))

def fix_pd_matrix(matrix, eps=1e-4):
    """
    Fix a positive definite matrix

    Args:
        matrix (np.array): the matrix to be fixed
        eps (float): the tolerance

    Returns:
        np.array: the fixed matrix
    """
    eigv, eigw= np.linalg.eigh(matrix)
    eigv = np.real(eigv)
    eigw = np.real(eigw)
    eigv[eigv <= eps]= eps
    matrix= np.dot(np.dot(eigw, np.diag(eigv)), eigw.T)
    return matrix

def reverse_matrix(matrix):
    """
    Reverse the order of eigenvalues.

    Args:
        matrix (np.array): the matrix to transform

    Returns:
        np.array: the transformed matrix
    """
    eigv, eigw = np.linalg.eigh(matrix)
    eigv = eigv[np.argsort(eigv)[len(eigv) - rankdata(eigv).astype(int)]]
    matrix = np.dot(np.dot(eigw, np.diag(eigv)), eigw.T)
    return matrix

def create_metric_tensor(n_dims, tu_indices, elements):
    """
    Create metric tensor from upper triangle matrix elements.

    Args:
         n_dims (int): dimensionality
         tu_indices (np.array): indices of the upper triangle
         elements (np.array): the values corresponding to the indices

    Returns:
        np.array: the constructed distance matrix
    """
    # creating the metric tensor
    dist= np.zeros((n_dims, n_dims))
    # injecting the regressed coefficients into the upper triangle
    dist[tu_indices]= elements
    # copying by transposing
    dist= (dist + dist.T)
    # halving the elements in the main diagonal due to duplication
    dist[np.diag_indices(n_dims)]= np.diag(dist)/2
    # creating a valid positive definite matrix
    dist= fix_pd_matrix(dist)

    return dist


def construct_tensor(X, dissim_matrix):
    """
    Construct tensor from dissimilarity matrix.

    Args:
        X (np.array): the data to use
        dissim_matrix (np.array): the dissimilarity matrix

    Returns:
        np.array: the metric tensor
    """

    # pre-calculating some triangle indices
    X_tu_indices= np.triu_indices(X.shape[0], k=0)
    d_tu_indices_0= np.triu_indices(X.shape[1], k=0)
    d_tu_indices_1= np.triu_indices(X.shape[1], k=1)

    n_upper, n_d= len(X_tu_indices[0]), len(d_tu_indices_0[0])

    # calculating the cross differences and extracting the upper triangle into
    # a row-major representation
    cross_diff_all= (X[:,None] - X)[X_tu_indices]

    # if the total number of pairs with distances is much greater than the
    # number of free components of the metric tensor, the samples prepared for
    # regression are sampled for efficiency
    mask= np.repeat(True, len(cross_diff_all))
    if n_upper > n_d*100:
        rng = np.arange(n_upper)
        mask[np.random.RandomState(5).permutation(rng)[n_d*100:]]= False

    # preparing the dissimilarity values for regression
    y_target= dissim_matrix[(X_tu_indices[0][mask], X_tu_indices[1][mask])]**2

    # calculating the cross product of components for each pair in the cross
    # difference matrix
    cross_diff_all= cross_diff_all[mask]
    cross_diff_cross_products= np.einsum('...i,...j->...ij',
                                        cross_diff_all,
                                        cross_diff_all)

    # adjustment due to extracting the upper triangle only
    cross_diff_cross_products[:, d_tu_indices_1[0], d_tu_indices_1[1]]*= 2

    # preparing the components of the cross products of pairwise distances
    # for regression
    X_target= cross_diff_cross_products[:, d_tu_indices_0[0], d_tu_indices_0[1]]

    # calculating the elements of the metric tensor by regression
    linearr= LinearRegression(fit_intercept=False).fit(X_target, y_target)

    # creating the metric tensor
    metric_tensor= create_metric_tensor(X.shape[1],
                                        d_tu_indices_0,
                                        linearr.coef_)

    return metric_tensor, linearr.score(X_target, y_target)


class ClassifierImpliedDissimilarityMatrix:
    """
    Computes classifier implied dissimilarity matrix
    """
    def __init__(self,
                 classifier=('sklearn.ensemble',
                            'RandomForestClassifier'),
                 classifier_params=None):
        """
        Constructor of the object

        Args:
            classifier (str): name of a classifier class (available in sklearn)
            classifier_params (dict): parameters of the classifier
        """
        self.classifier= classifier
        if classifier_params is None:
            classifier_params = {'n_estimators': 100,
                                    'min_samples_leaf': 2,
                                    'random_state': 5}
        self.classifier_params= classifier_params
        self.classifier_obj = None
        self.terminal_nodes = None

    def get_params(self):
        """
        The parameters of the object.

        Returns:
            dict: the parameters of the object
        """
        return {'classifier': self.classifier,
                'classifier_params': self.classifier_params}

    def fit(self, X, y):
        """
        Fit the object.

        Args:
            X (np.array): the X vectors
            y (np.array): the target vectors

        Returns:
            ClassifierImpliedDissimilarityMatrix: the fitted object
        """
        _logger.info("%s: fitting", self.__class__.__name__)
        self.classifier_obj = instantiate_obj((self.classifier[0],
                                                self.classifier[1],
                                                self.classifier_params))
        self.classifier_obj.fit(X, y)
        return self

    def transform_kneighbors(self, X):
        """
        Transform using kneighbors.

        Args:
            X (np.array): the vectors to transform

        Returns:
            np.array: the transformed vectors
        """
        _logger.info("%s: transform kneighbors", self.__class__.__name__)
        tnodes= self.classifier_obj.apply(X)
        tmp_nodes= np.vstack([self.terminal_nodes, tnodes])

        def func(vector):
            return 1*np.equal.outer(vector,
                                    vector)[-len(tnodes):][:, :-len(tnodes)]

        results= 1.0 - np.apply_along_axis(func, 0, tmp_nodes)
        results = results.sum(axis=2)/tmp_nodes.shape[1]

        return results

    def dissimilarity_matrix(self, X):
        """
        Calculates the dissimilarity matrix, if y is None, then uses the
        already fitted model. First needs to be called with valid y vector.

        Args:
            X (np.ndarray): explanatory variables
            y (np.array): target (class labels)
        """
        # terminal nodes: rows - samples, columns - trees in the forest
        self.terminal_nodes= self.classifier_obj.apply(X)

        def func(vector):
            return 1*np.equal.outer(vector, vector)

        result =  1.0 - np.apply_along_axis(func,
                                            0,
                                            self.terminal_nodes)
        return result.sum(axis=2)/self.terminal_nodes.shape[1]


class ClosestNeighborsInClasses:
    """
    Closest neighbors within class.
    """
    def __init__(self, n_neighbors=5):
        """
        Constructor of the class, determines the border points.

        Args:
            n_neighbors (int): number of neighbors
        """
        self.n_neighbors= n_neighbors

    def get_params(self):
        """
        Get the dict of parameters

        Returns:
            dict: the dict of parameters
        """
        return {'n_neighbors': self.n_neighbors}

    def fit_transform(self, X, y):
        """
        Fit and transform the dataset

        Args:
            X (np.array): the vectors to operate on
            y (np.array): the corresponding labels

        Returns:
            np.array, np.array: the border points
        """
        X_min= X[y == 1]
        X_maj= X[y == 0]

        nearestn= NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        _, ind_min= nearestn.kneighbors(X_min)
        _, ind_maj= nearestn.kneighbors(X_maj)

        label_min= np.all((y[ind_min] == 1), axis=1)
        label_maj= np.all((y[ind_maj] == 0), axis=1)

        X_final = np.vstack([X_maj[~label_maj], X_min[~label_min]])
        y_final = np.hstack([np.repeat(0, int(np.sum(label_maj))),
                             np.repeat(1, int(np.sum(label_min)))])

        return X_final, y_final


class RemoveCorrelatedColumns:
    """
    Transform the data by removing correlated columns
    """
    def __init__(self, threshold=0.99):
        """
        Constructor of the object

        Args:
            threshold (float): the correlation absolute value threshold
        """
        self.threshold = threshold
        self.remove_mask = None

    def fit(self, X):
        """
        Fit the transformer.

        Args:
            X (np.array): the vectors

        Returns:
            RemoveCorrelatedColumns: the fitted object
        """
        corr = np.abs(np.corrcoef(X.T))
        corr[np.tril_indices(len(corr), k=0)] = np.nan
        self.remove_mask = ~np.any(corr > self.threshold, axis=1)
        return self

    def transform(self, X):
        """
        Transform data.

        Args:
            X (np.array): the vectors to transform

        Returns:
            X (np.array): the transformed data
        """
        return X[:, self.remove_mask]

def discrete_variable_mask(X, threshold=5):
    """
    Determine the mask of discrete variables.

    Args:
        X (np.array): the data
        threshold (int): threshold on unique attribute elements

    Returns:
        np.array: the binary mask of discrete variable columns
    """
    return np.apply_along_axis(lambda x: len(np.unique(x)), 0, X) <= threshold

def estimate_mutual_information(X,
                                y,
                                normalize=True,
                                n_repeats= 10,
                                mi_params=None):
    """
    Estimate the mutual information between variables and a target.

    Args:
        X (np.array): explanatory variables
        y (np.array): the target variables
        normalize (bool): whether to normalize the results
        n_repeats (int): the number of repetitions
        mi_params (dict): parameters of the MI estimation

    Returns:
        np.array: the array of MI scores for each column of X
    """
    if mi_params is None:
        mi_params = {'n_neighbors': 3}

    discrete_mask= discrete_variable_mask(X, threshold=5)

    mutuali = [mutual_info_regression(X, y,
                                **mi_params,
                                discrete_features=discrete_mask,
                                random_state=j) for j in range(n_repeats)]

    mutuali = np.mean(np.array(mutuali), axis=0)

    if normalize:
        mutuali = mutuali / np.mean(mutuali)

    return mutuali

def n_neighbors_func(X_base,
                        X_neighbors=None,
                        n_neighbors=5,
                        metric_tensor=None,
                        return_distance=False):
    """
    Determines the n closest neighbors with a metric tensor.

    Args:
        X_base (np.array): the base data
        X_neighbors (np.array): the neighbors data
        n_neighbors (int): the number of neighbors
        metric_tensor (np.array): the tensor to be used
        return_distance (bool): whether to return the distance

    Returns:
        np.array(, np.array): the indices of closest neighbors and optionally
                                their distances
    """
    if metric_tensor is None:
        metric_tensor = np.eye(X_base.shape[1])

    X_neighbors= X_neighbors if X_neighbors is not None else X_base

    X_diff= (X_base[:,None] - X_neighbors)

    distm= np.sqrt(np.einsum('ijk,ijk -> ij',
                            X_diff,
                            np.dot(X_diff, metric_tensor)).T).T

    results_ind= np.apply_along_axis(np.argsort,
                                        axis=1,
                                        arr=distm)[:,:(n_neighbors)]

    if not return_distance:
        return results_ind

    distances = distm[np.arange(distm.shape[0])[:,None], results_ind]
    return distances, results_ind


def psd_mean(matrices):
    """
    Estimate the mean of positive semi-definite matrices

    Args:
        matrices (iterable): the matrices

    Returns:
        np.array: the estimated mean matrix
    """
    result= np.mean(matrices, axis=0)

    result= fix_pd_matrix(result)

    return result

class MetricTensor:
    """
    Class representing a metric tensor.
    """

    def __init__(self,
                 metric_learning_method=None,
                 **_kwargs):
        """
        MetricTensor constructor

        Args:
            metric_learning_method (str): metric learning algorithm

        """
        self.metric_learning_method = metric_learning_method
        self.metric_tensor = None

    def get_params(self):
        """
        Get the paraemters of the object.

        Returns:
            dict: the parameters of the object
        """
        return {'metric_learning_method': self.metric_learning_method}

    def _train_metric_learning(self, X, y, method='NCA', *,
                                random_state=None, prior='identity'):
        """
        Do the metric learning training

        Args:
            X (np.array): the explanataory variables
            y (np.array): the target variable
            method (str): metric learning method
            random_state (int/None): random state to pass

        Returns:
            np.array: the metric tensor
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == 'ITML':
                return ITML_Supervised(random_state=random_state,
                                        prior=prior).fit(X, y)\
                                        .get_mahalanobis_matrix()
            if method == 'LSML':
                return LSML_Supervised().fit(X, y)\
                                        .get_mahalanobis_matrix()

            return NCA().fit(X, y)\
                                .get_mahalanobis_matrix()

    def tensor(self, X, y):
        """
        Create metric tensor for the parameters.

        Args:
            X (np.array): the explanataory variables
            y (np.array): the target variable

        Returns:
            np.array: the metric tensor
        """
        X_mod, index = np.unique(X, axis=0, return_index=True)
        y_mod = y[index]

        _logger.info("%s: executing metric learning with %s",
                        self.__class__.__name__,
                        self.metric_learning_method)

        if self.metric_learning_method == 'ITML':
            self.metric_tensor = self._train_metric_learning(X_mod,
                                                y_mod,
                                                self.metric_learning_method)
        elif self.metric_learning_method == 'rf':
            dissim= ClassifierImpliedDissimilarityMatrix().fit(X, y)\
                                            .dissimilarity_matrix(X)
            self.metric_tensor, _ = construct_tensor(X, dissim)
        elif self.metric_learning_method == 'LSML':
            self.metric_tensor = self._train_metric_learning(X_mod,
                                                y_mod,
                                                self.metric_learning_method)
        elif self.metric_learning_method == 'cov':
            self.metric_tensor = np.linalg.inv(fix_pd_matrix(np.cov(X.T)))
        elif self.metric_learning_method == 'cov_min':
            cov= np.cov(X[y == 1].T)
            self.metric_tensor = np.linalg.inv(fix_pd_matrix(cov))
        elif self.metric_learning_method == 'MI_weighted':
            mutuali= estimate_mutual_information(X, y)
            self.metric_tensor= np.diag(mutuali)
        elif self.metric_learning_method == 'id':
            self.metric_tensor= np.eye(len(X_mod[0]))
        elif self.metric_learning_method == 'ITML_mi':
            self.metric_tensor = self._train_metric_learning(X_mod,
                                                y_mod,
                                                self.metric_learning_method)
            mutuali= estimate_mutual_information(X, y)
            self.metric_tensor= np.matmul(self.metric_tensor, np.diag(mutuali))
        elif self.metric_learning_method == 'NCA':
            self.metric_tensor= NCA().fit(X_mod, y_mod)\
                                        .get_mahalanobis_matrix()
        elif self.metric_learning_method == 'gmean':
            matrices = [self._train_metric_learning(X_mod,
                                                y_mod,
                                                self.metric_learning_method,
                                                prior='random') for i in range(2)]

            self.metric_tensor= psd_mean(matrices)

        return self.metric_tensor

class MetricLearningMixin:
    """
    Mixin class for oversampling methods supporting custom metrics
    """

    def metric_tensor_from_nn_params(self, nn_params, X, y):
        """
        Determine the metric tensor from the given parameters.

        Args:
            nn_params (dict): the nearest neighbors parameters
            X (np.array): the explanatory variables
            y (np.array): the target variables

        Returns:
            np.array/None: the metric tensor
        """
        if nn_params.get('metric', None) == 'precomputed' \
                and nn_params.get('metric_tensor', None) is None:
            return MetricTensor(**nn_params).tensor(X, y)

        if nn_params.get('metric_tensor', None) is not None:
            return nn_params['metric_tensor']

        return None

    def get_params(self):
        """
        Get the parameters of the object.

        Returns:
            dict: the parameter dict
        """
        return {}

class NearestNeighborsWithMetricTensor:
    """
    NearestNeighbors driven by a metric tensor
    """
    def __init__(self,
                 n_neighbors=5,
                 *,
                 radius=1.0,
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2, # pylint: disable=invalid-name
                 metric_params=None,
                 metric_tensor=None,
                 n_jobs=None,
                 **_kwargs):
        """
        Constructor of the class

        Args:
            n_neighbors (int): the number of neighbors
            radius (float): the radius of neighbors
            algorithm (str): the algorithm to be used
            leaf_size (int): the size of leafs for quadtree estimation
            metric (str): the metric to be used
            p (int): the p-norm p value
            metric_params (dict): the parameters for metric learning
            metric_tensor (np.array): the metric tensor to be used
            n_jobs (int): the number of jobs
        """
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.metric = metric
        self.metric_tensor = metric_tensor
        self.X_fitted = None

        self.nearestn= NearestNeighbors(n_neighbors=n_neighbors,
                                        radius=radius,
                                        algorithm=algorithm,
                                        leaf_size=leaf_size,
                                        metric=metric,
                                        p=p,
                                        metric_params=metric_params,
                                        n_jobs=n_jobs)

    def get_params(self):
        """
        Get the parameters of the object.

        Returns:
            dict: the parameters of the object
        """
        return {**self.nearestn.get_params(),
                "metric_tensor": self.metric_tensor}

    def fit(self, X):
        """
        Fit the nearest neighbors

        Args:
            X (np.array): the vectors to fit

        Returns:
            NearestNeighborsWithMetricTensor: the fitted object
        """
        _logger.info("%s: NN fitting with metric %s",
                        self.__class__.__name__,
                        self.metric)
        if self.metric != 'precomputed' or self.metric_tensor is None:
            self.nearestn.fit(X)
        else:
            self.X_fitted= X

        return self

    def kneighbors(self,
                    X=None,
                    n_neighbors=None,
                    return_distance=True):
        """
        Determine the k nearest neighbors

        Args:
            X (np.array): the vectors to determine the neighbors for
            n_neighbors (int): the number of neighbors
            return_distance (bool): whether to return the distance

        Returns:
            np.array(, np.array): the indices of nearest neighbors and
                                    optionally the distances
        """
        _logger.info("%s: kneighbors query %s",
                        self.__class__.__name__,
                        self.metric)

        if self.metric != 'precomputed' or self.metric_tensor is None:
            return self.nearestn.kneighbors(X, n_neighbors, return_distance)

        n_neighbors = coalesce(n_neighbors, self.n_neighbors)
        return n_neighbors_func(X, self.X_fitted,
                            metric_tensor=self.metric_tensor,
                            n_neighbors=n_neighbors,
                            return_distance=return_distance)

    def radius_neighbors(self,
                            X=None,
                            radius=None,
                            return_distance=True,
                            sort_results=False):
        """
        Determine the neighbors within a given radius.

        Args:
            X (np.array): the vectors to query the neighbors of
            radius (float/None): the radius
            return_distance (bool): whether to return the distances
            sort_results (bool): whether to sort the results
        """
        _logger.info("%s: radius neighbors query %s",
                        self.__class__.__name__,
                        self.metric)

        if self.metric != 'precomputed' or self.metric_tensor is None:
            return self.nearestn.radius_neighbors(X,
                                                    radius,
                                                    return_distance,
                                                    sort_results)

        diffs = (self.X_fitted[:,None] - X)
        distm = np.einsum('ijk,ijk -> ij',
                            diffs,
                            np.dot(diffs, self.metric_tensor)).T
        distm = np.sqrt(distm)

        results_dist= []
        results_ind= []

        for _, row in enumerate(distm):
            mask= np.where(row <= radius)[0]
            results_dist.append(row[mask])
            results_ind.append(mask)

        results_dist = np.array(results_dist, dtype=object)
        results_ind = np.array(results_ind, dtype=object)

        if return_distance:
            return results_dist, results_ind

        return results_ind
