import numpy as np

from .._metric_tensor import NearestNeighborsWithMetricTensor, MetricTensor
from ._OverSampling import OverSampling
from .._logger import logger
_logger= logger

__all__= ['SPY']

class SPY(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{spy,
                            author={Dang, X. T. and Tran, D. H. and Hirose, O.
                                    and Satou, K.},
                            booktitle={2015 Seventh International Conference
                                        on Knowledge and Systems Engineering
                                        (KSE)},
                            title={SPY: A Novel Resampling Method for
                                    Improving Classification Performance in
                                    Imbalanced Data},
                            year={2015},
                            volume={},
                            number={},
                            pages={280-285},
                            keywords={decision making;learning (artificial
                                        intelligence);pattern classification;
                                        sampling methods;SPY;resampling
                                        method;decision-making process;
                                        biomedical data classification;
                                        class imbalance learning method;
                                        SMOTE;oversampling method;UCI
                                        machine learning repository;G-mean
                                        value;borderline-SMOTE;
                                        safe-level-SMOTE;Support vector
                                        machines;Training;Bioinformatics;
                                        Proteins;Protein engineering;Radio
                                        frequency;Sensitivity;Imbalanced
                                        dataset;Over-sampling;
                                        Under-sampling;SMOTE;
                                        borderline-SMOTE},
                            doi={10.1109/KSE.2015.24},
                            ISSN={},
                            month={Oct}}
    """

    categories = [OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 n_neighbors=5,
                 *,
                 nn_params={},
                 threshold=0.5,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            n_neighbors (int): number of neighbors in nearest neighbor
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any 
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            threshold (float): threshold*n_neighbors gives the threshold z
                                described in the paper
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_in_range(threshold, "threshold", [0, 1])
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self.threshold = threshold
        self.n_jobs = n_jobs

        # random state takes no effect for this technique
        self.set_random_state(random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'n_neighbors': [3, 5, 7],
                                  'threshold': [0.3, 0.5, 0.7]}
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

        X_min = X[y == self.min_label]

        # fitting nearest neighbors model
        n_neighbors = min([len(X), self.n_neighbors + 1])

        nn_params= {**self.nn_params}
        nn_params['metric_tensor']= self.metric_tensor_from_nn_params(nn_params, X, y)

        nn = NearestNeighborsWithMetricTensor(n_neighbors=n_neighbors, 
                                                n_jobs=self.n_jobs, 
                                                **(nn_params))
        nn.fit(X)
        ind = nn.kneighbors(X_min, return_distance=False)

        y_new = y.copy()
        z = self.threshold*n_neighbors

        # checking the neighbors of each minority sample
        for i in range(len(X_min)):
            majority_mask = y[ind[i][1:]] == self.maj_label
            x = np.sum(majority_mask)
            # if the number of majority samples in the neighborhood is
            # smaller than a threshold
            # their labels are changed to minority
            if x < z:
                y_new[ind[i][1:][majority_mask]] = self.min_label

        return X.copy(), y_new

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'n_neighbors': self.n_neighbors,
                'nn_params': self.nn_params,
                'threshold': self.threshold,
                'n_jobs': self.n_jobs}
