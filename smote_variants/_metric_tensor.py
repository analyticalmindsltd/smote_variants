import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from metric_learn import ITML_Supervised, LSML_Supervised

from ._logger import logger
_logger = logger

__all__= ['NearestNeighborsWithClassifierDissimilarity',
          'NearestNeighborsWithMetricTensor',
          'ClassifierImpliedDissimilarityMatrix',
          'MetricTensor',
          'MetricLearningMixin',
          'pairwise_distances_mahalanobis',
          'generate_samples',
          'AdditionalItems']


def pairwise_distances_mahalanobis(X, Y=None, tensor=None):
    if Y is None:
        Y = X
    if tensor is None:
        tensor = np.eye(len(X[0]))
    tmp= (X[:,None] - Y)
    return np.sqrt(np.einsum('ijk,ijk -> ij', tmp, np.dot(tmp, tensor)).T)


def fix_pd_matrix(M, eps=1e-4):
    eigv, eigw= np.linalg.eigh(M)
    eigv[eigv<=eps]= eps
    M= np.dot(np.dot(eigw, np.diag(eigv)), eigw.T)
    return M


def create_metric_tensor(d, tu_indices, elements):
    # creating the metric tensor
    dist= np.zeros((d, d))
    # injecting the regressed coefficients into the upper triangle
    dist[tu_indices]= elements
    # copying by transposing
    dist= (dist + dist.T)
    # halving the elements in the main diagonal due to duplication
    dist[np.diag_indices(d)]= np.diag(dist)/2
    # creating a valid positive definite matrix
    dist= fix_pd_matrix(dist)
    
    return dist


def construct_tensor(X, dissim_matrix):
    # extracting some constants
    n, d= len(X), len(X[0])
    
    # pre-calculating some triangle indices
    X_tu_indices= np.triu_indices(n, k=0)
    d_tu_indices_0= np.triu_indices(d, k=0)
    d_tu_indices_1= np.triu_indices(d, k=1)
    
    n_upper, n_d= len(X_tu_indices[0]), len(d_tu_indices_0[0])
    
    # calculating the cross differences and extracting the upper triangle into
    # a row-major representation
    cross_diff_all= (X[:,None] - X)[X_tu_indices]

    # if the total number of pairs with distances is much greater than the
    # number of free components of the metric tensor, the samples prepared for
    # regression are sampled for efficiency
    mask= np.repeat(True, len(cross_diff_all))
    if n_upper > n_d*100:
        mask[np.random.RandomState(5).permutation(np.arange(n_upper))[n_d*100:]]= False
    
    # preparing the dissimilarity values for regression
    cy= dissim_matrix[(X_tu_indices[0][mask], X_tu_indices[1][mask])]**2

    # calculating the cross product of components for each pair in the cross
    # difference matrix
    cross_diff_sampled= cross_diff_all[mask]
    cross_diff_cross_products= np.einsum('...i,...j->...ij', 
                                        cross_diff_sampled, 
                                        cross_diff_sampled)

    # adjustment due to extracting the upper triangle only
    cross_diff_cross_products[:, d_tu_indices_1[0], d_tu_indices_1[1]]*= 2

    # preparing the components of the cross products of pairwise distances for regression
    cX= cross_diff_cross_products[:, d_tu_indices_0[0], d_tu_indices_0[1]]

    # calculating the elements of the metric tensor by regression
    lr= LinearRegression(fit_intercept=False).fit(cX, cy)
    #print(lr.score(cX, cy))
    r2= lr.score(cX, cy)
    
    metric_elements= lr.coef_
    
    # creating the metric tensor
    metric_tensor= create_metric_tensor(d, d_tu_indices_0, metric_elements)
    
    return metric_tensor, r2
        
        
class ClassifierImpliedDissimilarityMatrix:
    """
    Computes classifier implied dissimilarity matrix
    """
    def __init__(self,
                 classifier='RandomForestClassifier',
                 classifier_params={'n_estimators': 100,
                                    'min_samples_leaf': 2,
                                    'random_state': 5}):
        """
        Constructor of the object
        
        Args:
            classifier (str): name of a classifier class (available in sklearn)
            classifier_params (dict): parameters of the classifier
        """
        self.classifier= classifier
        self.classifier_params= classifier_params
    
    def get_params(self):
        return {'classifier': self.classifier,
                'classifier_params': self.classifier_params}
    
    def fit(self, X, y):
        _logger.info(self.__class__.__name__ + ": " +
                     "fitting")
        self.classifier_obj= eval(self.classifier)(**self.classifier_params).fit(X, y)
        return self

    def transform_fit(self, X):
        _logger.info(self.__class__.__name__ + ": " +
                     "transform_fit")
        return self.dissimilarity_matrix(X)

    def transform_kneighbors(self, X):
        _logger.info(self.__class__.__name__ + ": " +
                     "transform_kneithbors")
        tnodes= self.classifier_obj.apply(X)
        tmp_nodes= np.vstack([self.terminal_nodes, tnodes])
        
        results= 1.0 - np.apply_along_axis(lambda x: 1*np.equal.outer(x, x)[-len(tnodes):][:, :-len(tnodes)], 
                                    0, 
                                    tmp_nodes).sum(axis=2)/tmp_nodes.shape[1]
        
        return results
    
    def dissimilarity_matrix(self, X):
        """
        Calculates the dissimilarity matrix, if y is None, then uses the already fitted
        model. First needs to be called with valid y vector.
        
        Args:
            X (np.ndarray): explanatory variables
            y (np.array): target (class labels)
        """
        # terminal nodes: rows - samples, columns - trees in the forest
        self.terminal_nodes= self.classifier_obj.apply(X)
        
        return 1.0 - np.apply_along_axis(lambda x: 1*np.equal.outer(x, x), 
                                    0, 
                                    self.terminal_nodes).sum(axis=2)/self.terminal_nodes.shape[1]


class ClosestNeighborsInClasses:
    def __init__(self, n_neighbors=5):
        self.n_neighbors= n_neighbors
    
    def fit_transform(self, X, y):
        X_min= X[y == 1]
        X_maj= X[y == 0]
        
        nn= NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        _, ind_min= nn.kneighbors(X_min)
        _, ind_maj= nn.kneighbors(X_maj)
        
        label_min= np.all((y[ind_min] == 1), axis=1)
        label_maj= np.all((y[ind_maj] == 0), axis=1)
        
        return np.vstack([X_maj[~label_maj], X_min[~label_min]]), np.hstack([np.repeat(0, np.sum(label_maj), np.repeat(1, np.sum(label_min)))])

class RemoveCorrelatedColumns:
    def __init__(self, threshold=0.99):
        self.threshold= threshold

    def fit(self, X):
        cc= np.abs(np.corrcoef(X.T))
        cc[np.tril_indices(len(cc), k=0)]= np.nan
        self.remove_mask= ~np.any(cc > self.threshold, axis=1)
        return self
    
    def transform(self, X):
        return X[:,self.remove_mask]

def discrete_variable_mask(X, threshold=5):
    return np.apply_along_axis(lambda x: len(np.unique(x)), 0, X) <= threshold

def estimate_mutual_information(X, 
                                y, 
                                normalize=True,
                                n_repeats= 10, 
                                mi_params={'n_neighbors': 3}):
    discrete_mask= discrete_variable_mask(X, threshold=5)
    mi= np.mean(np.array([mutual_info_regression(X, 
                                                y, 
                                                **mi_params, 
                                                discrete_features=discrete_mask,
                                                random_state=j) for j in range(5, n_repeats + 5)]), axis=0)
    if normalize:
        mi=mi/np.mean(mi)
    return mi

def n_neighbors_func(X0, X1=None, n_neighbors=5, metric_tensor=None, return_distance=False):
    metric_tensor= metric_tensor if metric_tensor is not None else np.eye(X0.shape[1])
    X1= X1 if X1 is not None else X0

    X_diffs= (X0[:,None] - X1)
    
    dm= np.sqrt(np.einsum('ijk,ijk -> ij', 
                            X_diffs, 
                            np.dot(X_diffs, metric_tensor)).T)

    results_ind= np.apply_along_axis(np.argsort, axis=1, arr=dm)[:,:(n_neighbors)]

    if not return_distance:
        return results_ind
    else:
        return dm[np.arange(dm.shape[0])[:,None], results_ind], results_ind

class MetricTensor:
    def __init__(self,
                 metric_learning_method='ITML',
                 **kwargs):
        """
        MetricTensor constructor

        Args:
            metric_learning_method (str): metric learning algorithm

        """
        self.metric_learning_method = metric_learning_method
    
    def tensor(self, X, y):
        X_mod, index= np.unique(X, axis=0, return_index=True)
        y_mod= y[index]

        _logger.info(self.__class__.__name__ + ": " +
                     "executing metric learning with %s" % self.metric_learning_method)

        if self.metric_learning_method == 'ITML':
            self.metric_tensor= ITML_Supervised().fit(X_mod, y_mod).get_mahalanobis_matrix()
        elif self.metric_learning_method == 'rf':
            dm= ClassifierImpliedDissimilarityMatrix().fit(X, y).dissimilarity_matrix(X)
            self.metric_tensor, _ = construct_tensor(X, dm)
        elif self.metric_learning_method == 'LSML':
            self.metric_tensor= LSML_Supervised().fit(X_mod, y_mod).get_mahalanobis_matrix()
            eigv, eigw = np.linalg.eigh(self.metric_tensor)
            self.metric_tensor = np.dot(np.dot(eigw, np.diag(sorted(eigv, reverse=True))), eigw.T)
        elif self.metric_learning_method == 'cov':
            cov= np.cov(X.T)
            self.metric_tensor = np.linalg.inv(cov)
        elif self.metric_learning_method == 'cov_min':
            cov= np.cov(X[y == 1].T)
            self.metric_tensor = np.linalg.inv(cov)
        elif self.metric_learning_method == 'MI_weighted':
            mi= estimate_mutual_information(X, y)
            self.metric_tensor= np.diag(mi)
        else:
            self.metric_tensor= None
        
        return self.metric_tensor

class MetricLearningMixin:
    def metric_tensor_from_nn_params(self, nn_params, X, y):
        if nn_params.get('metric', None) == 'precomputed' and nn_params.get('metric_tensor', None) is None:
            return MetricTensor(**nn_params).tensor(X, y)
        elif nn_params.get('metric_tensor', None) is not None:
            return nn_params['metric_tensor']
        return None

class NearestNeighborsWithMetricTensor:
    def __init__(self,
                 n_neighbors=5, 
                 radius=1.0, 
                 algorithm='auto', 
                 leaf_size=30, 
                 metric='minkowski', 
                 p=2, 
                 metric_params=None, 
                 metric_tensor=None,
                 n_jobs=None,
                 **kwargs):
        self.n_neighbors= n_neighbors
        self.radius= radius
        self.algorithm= algorithm
        self.leaf_size= leaf_size
        self.metric= metric
        self.p= p
        self.metric_params= metric_params
        self.metric_tensor= metric_tensor
        self.n_jobs= n_jobs
        
        if metric != 'precomputed':
            self.nn= NearestNeighbors(n_neighbors=n_neighbors,
                                        radius=radius,
                                        algorithm=algorithm,
                                        leaf_size=leaf_size,
                                        metric=metric,
                                        p=p,
                                        metric_params=metric_params,
                                        n_jobs=n_jobs)
            
    
    def fit(self, X):
        _logger.info(self.__class__.__name__ + ": " +
                     "NN fitting with metric %s" % self.metric)
        if self.metric != 'precomputed' or self.metric_tensor is None:
            self.nn.fit(X)
        else:
            self.X_fitted= X
        
        return self
    
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        _logger.info(self.__class__.__name__ + ": " +
                     "kneighbors query %s" % self.metric)
        if self.metric != 'precomputed' or self.metric_tensor is None:
            return self.nn.kneighbors(X, n_neighbors, return_distance)
        else:
            # calculating the distance matrix
            return n_neighbors_func(self.X_fitted, X, 
                               metric_tensor=self.metric_tensor, 
                               n_neighbors=self.n_neighbors,
                               return_distance=return_distance)

    
    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        _logger.info(self.__class__.__name__ + ": " +
                     "radius_neighbors query %s" % self.metric)
        if self.metric != 'precomputed' or self.metric_tensor is None:
            return self.nn.radius_neighbors(X, radius, return_distance, sort_results)
        else:
            # TODO: better implementation needed
            X= (self.X_fitted[:,None] - X)
            dm= np.sqrt(np.einsum('ijk,ijk -> ij', X, np.dot(X, self.metric_tensor)).T)
            results_dist= []
            results_ind= []
            
            for i in range(len(dm)):
                mask= np.where(dm[i] <= radius)[0]
                results_dist.append(dm[i][mask])
                results_ind.append(mask)

            if return_distance:
                return results_dist, results_ind
            else:
                return results_ind



class NearestNeighborsWithClassifierDissimilarity:
    def __init__(self,
                 n_neighbors=5, 
                 radius=1.0, 
                 algorithm='auto', 
                 leaf_size=30, 
                 metric='minkowski', 
                 p=2, 
                 metric_params=None, 
                 n_jobs=None,
                 metric_learning='ITML',
                 metric_tensor=None,
                 X=None,
                 y=None):
        self.n_neighbors= n_neighbors
        self.radius= radius
        self.algorithm= algorithm
        self.leaf_size= leaf_size
        self.metric= metric
        self.p= p
        self.metric_params= metric_params
        self.n_jobs= n_jobs
        self.metric_learning=metric_learning
        self.X= X
        self.y= y
        
        self.metric_tensor= metric_tensor
        
        if metric == 'precomputed' and self.metric_tensor is None:
            X_mod, index= np.unique(X, axis=0, return_index=True)
            y_mod= y[index]
            self.metric_tensor = MetricTensor(self.metric_learning, 
                                              self.metric).tensor(X_mod, y_mod)
        else:
            self.nn= NearestNeighbors(n_neighbors=n_neighbors,
                                                radius=radius,
                                                algorithm=algorithm,
                                                leaf_size=leaf_size,
                                                metric=metric,
                                                p=p,
                                                metric_params=metric_params,
                                                n_jobs=n_jobs)
            
    
    def fit(self, X):
        _logger.info(self.__class__.__name__ + ": " +
                     "NN fitting with metric %s" % self.metric)
        if self.metric != 'precomputed':
            self.nn.fit(X)
        else:
            self.X= X
        
        return self
    
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        _logger.info(self.__class__.__name__ + ": " +
                     "kneighbors query %s" % self.metric)
        if self.metric != 'precomputed':
            return self.nn.kneighbors(X, n_neighbors, return_distance)
        else:
            # calculating the distance matrix
            return n_neighbors_func(self.X, X, 
                               metric_tensor=self.metric_tensor, 
                               n_neighbors=self.n_neighbors,
                               return_distance=return_distance)

    
    def radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False):
        _logger.info(self.__class__.__name__ + ": " +
                     "radius_neighbors query %s" % self.metric)
        if self.metric != 'precomputed':
            return self.nn.radius_neighbors(X, radius, return_distance, sort_results)
        else:
            # TODO: better implementation needed
            X= (self.X[:,None] - X)
            dm= np.sqrt(np.einsum('ijk,ijk -> ij', X, np.dot(X, self.metric_tensor)).T)
            results_dist= []
            results_ind= []
            
            for i in range(len(dm)):
                mask= np.where(dm[i] <= radius)[0]
                results_dist.append(dm[i][mask])
                results_ind.append(mask)

            if return_distance:
                return results_dist, results_ind
            else:
                return results_ind

def generate_samples(X_base, 
                     X_neighbors, 
                     random_offsets, 
                     weights=None,
                     weights_transform=lambda x: x,
                     distance_fraction=0.0,
                     random_state=5):
    print('distance_fraction', distance_fraction)
    if weights is None:
        results= X_base + random_offsets[:,None]*(X_neighbors - X_base)
    if type(weights) is np.ndarray:
        weights= weights/np.max(weights)
        weights= weights_transform(weights)
        #print('aaa', weights)
        results= X_base + random_offsets[:,None]*weights*(X_neighbors - X_base)
        #results2= X_base + random_offsets[:,None]*(X_neighbors - X_base)
        #print('aaa', random_offsets[0])
        #print('bbb', results[0])
        #print('ccc', results2[0])
        #print('ddd', X_neighbors[0])
        #print('eee', X_base[0])
    
    if distance_fraction > 0.0:
        weights= weights_transform(weights)
        diffs= X_neighbors - X_base
        norms= np.linalg.norm(diffs, axis=1)
        rnd= np.random.RandomState(random_state)
        indices= rnd.choice(np.arange(X_base.shape[1]), (2, diffs.shape[0]), replace=True)
        normal_vectors= np.zeros(shape=X_base.shape)
        normal_vectors[:,indices[:,0]]= -diffs[:,indices[:,1]]
        normal_vectors[:,indices[:,1]]= diffs[:,indices[:,0]]
        
        norms_new= np.linalg.norm(normal_vectors, axis=1)
        norms_new[np.abs(norms_new) < 1e-5]= 1.0
        
        #print(distance_fraction, np.std(norms), np.std(norms_new))
        normal_vectors= (normal_vectors.T*(1.0/norms_new*norms*distance_fraction)).T
        print(distance_fraction)
        #print(results[0])
        results+= normal_vectors
        #results+= np.random.rand(results.shape[0], results.shape[1])*10
        #print(results[0])
    
    return results
        
class AdditionalItems:
    def __init__(self, nn_params, sampling_params):
        self.nn_params= nn_params.copy()
        self.sampling_params= sampling_params.copy()
    
    def fit(self, X, y):
        if not 'metric_tensor' in self.nn_params:
            self.nn_params['metric_tensor']= MetricTensor(**self.nn_params).tensor(X, y)
        
        if self.sampling_params['weights'] == 'metric_tensor' and self.nn_params['metric_tensor'] is not None:
            self.sampling_params['weights']= self.nn_params['metric_tensor']
        elif self.sampling_params['weights'] == 'metric_tensor' and self.nn_params['metric_tensor'] is None:
            self.sampling_params['weights']= np.diag(MetricTensor(**self.sampling_params).tensor(X, y))
        
        if 'metric' in self.sampling_params:
            del self.sampling_params['metric']
        if 'metric_learning' in self.sampling_params:
            del self.sampling_params['metric_learning']
        
        return self
    