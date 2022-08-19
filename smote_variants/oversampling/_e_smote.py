"""
This module implements the E_SMOTE method.
"""

import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..base import coalesce, coalesce_dict
from ..base import OverSampling
from ._smote import SMOTE

from .._logger import logger
_logger= logger

__all__= ['E_SMOTE']

class E_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{e_smote,
                            author={Deepa, T. and Punithavalli, M.},
                            booktitle={2011 3rd International Conference on
                                        Electronics Computer Technology},
                            title={An E-SMOTE technique for feature selection
                                    in High-Dimensional Imbalanced Dataset},
                            year={2011},
                            volume={2},
                            number={},
                            pages={322-324},
                            keywords={bioinformatics;data mining;pattern
                                        classification;support vector machines;
                                        E-SMOTE technique;feature selection;
                                        high-dimensional imbalanced dataset;
                                        data mining;bio-informatics;dataset
                                        balancing;SVM classification;micro
                                        array dataset;Feature extraction;
                                        Genetic algorithms;Support vector
                                        machines;Data mining;Machine learning;
                                        Bioinformatics;Cancer;Imbalanced
                                        dataset;Featue Selection;E-SMOTE;
                                        Support Vector Machine[SVM]},
                            doi={10.1109/ICECTECH.2011.5941710},
                            ISSN={},
                            month={April}}

    Notes:
        * This technique is basically unreproducible. I try to implement
            something following the idea of applying some simple genetic
            algorithm for optimization.
        * In my best understanding, the technique uses evolutionary algorithms
            for feature selection and then applies vanilla SMOTE on the
            selected features only.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_dim_reduction,
                  OverSampling.cat_memetic,
                  OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 n_neighbors=5,
                 *,
                 nn_params=None,
                 ss_params=None,
                 min_features=2,
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
            n_neighbors (int): number of neighbors in the nearest neighbors
                                component
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            ss_params (dict): simplex sampling parameters
            min_features (int): minimum number of features
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'random',
                            'gaussian_component': None}

        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(n_neighbors, "n_neighbors", 1)
        self.check_greater_or_equal(min_features, "min_features", 1)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.nn_params = coalesce(nn_params, {})
        self.ss_params = coalesce_dict(ss_params, ss_params_default)
        self.min_features = min_features
        self.n_jobs = n_jobs

        self.mask = None

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'n_neighbors': [3, 5, 7],
                                  'min_features': [1, 2, 3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def crossover(self, mask_a, mask_b, min_features):
        """
        Crossover operation for two masks

        Args:
            mask_a (np.array): binary mask 1
            mask_b (np.array): binary mask 2

        Returns:
            np.array: the result of crossover
        """
        mask = mask_a.copy()
        for idx, flag in enumerate(mask_b):
            if self.random_state.randint(0, 2) == 0:
                mask[idx] = flag

        while np.sum(mask) < min_features:
            mask[self.random_state.randint(len(mask))] = True

        return mask

    def mutate(self, mask_old, min_features):
        """
        Mutation operation for a mask

        Args:
            mask_old (np.array): binary mask

        Returns:
            np.array: the result of mutation
        """
        mask = mask_old.copy()
        for idx, flag in enumerate(mask):
            if self.random_state.randint(0, 2) == 0:
                mask[idx] = not flag

        while np.sum(mask) < min_features:
            mask[self.random_state.randint(len(mask))] = True

        return mask

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        min_features = min(self.min_features, len(X[0]))

        if len(X) < 800:
            classifier = SVC(gamma='auto', random_state=self._random_state_init)
        else:
            classifier = DecisionTreeClassifier(
                max_depth=4, random_state=self._random_state_init)

        # parameters of the evolutionary algorithm
        n_generations = 50
        n_population = 5

        # creating initial mask
        mask = np.repeat([False], X.shape[1])
        mask[np.random.choice(np.arange(mask.shape[0]), 1)] = True

        # generating initial population
        population = [[0, mask.copy()] for _ in range(n_population)]
        for _ in range(n_generations):
            # in each generation
            for _ in range(n_population):
                # for each element of a population
                if self.random_state.randint(0, 2) == 0:
                    # crossover
                    i_0 = self.random_state.randint(n_population)
                    i_1 = self.random_state.randint(n_population)
                    mask = self.crossover(population[i_0][1],
                                            population[i_1][1],
                                            min_features)
                else:
                    # mutation
                    idx = self.random_state.randint(n_population)
                    mask = self.mutate(population[idx][1],
                                        min_features)
                # evaluation
                _logger.info("%s evaluting mask selection with features %d/%d",
                            self.__class__.__name__, np.sum(mask), len(mask))
                classifier.fit(X[:, mask], y)
                score = np.sum(y == classifier.predict(X[:, mask]))/len(y)
                # appending the result to the population
                population.append([score, mask])
            # sorting the population in a reversed order and keeping the
            # elements with the highest scores
            population = sorted(population, key=lambda x: -x[0])[:n_population]

        self.mask = population[0][1]
        # resampling the population in the given dimensions

        smote = SMOTE(proportion=self.proportion,
                      n_neighbors=self.n_neighbors,
                      nn_params=self.nn_params,
                      ss_params=self.ss_params,
                      n_jobs=self.n_jobs,
                      random_state=self._random_state_init)

        return smote.sample(X[:, self.mask], y)

    def preprocessing_transform(self, X):
        """
        Transform new data by the learnt transformation

        Args:
            X (np.array): new data

        Returns:
            np.array: transformed data
        """
        return X[:, self.mask]

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'n_neighbors': self.n_neighbors,
                'min_features': self.min_features,
                'nn_params': self.nn_params,
                'ss_params': self.ss_params,
                'n_jobs': self.n_jobs,
                **OverSampling.get_params(self)}
