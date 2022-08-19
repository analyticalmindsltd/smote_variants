"""
This module implements the polynom_fit_SMOTE method.
"""

import numpy as np

from ..base import coalesce_dict
from ..base import OverSampling, OverSamplingSimplex

from .._logger import logger
_logger= logger

__all__= ['polynom_fit_SMOTE_star',
            'polynom_fit_SMOTE_bus',
            'polynom_fit_SMOTE_poly',
            'polynom_fit_SMOTE_mesh']

class polynom_fit_SMOTE(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 topology='star',
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            topoplogy (str): 'star'/'bus'/'poly_N'
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        if topology.startswith('poly'):
            self.check_greater_or_equal(
                int(topology.split('_')[-1]), 'topology', 1)
        else:
            self.check_isin(topology, "topology", ['star', 'bus'])

        self.proportion = proportion
        self.topology = topology

    #@ classmethod
    #def parameter_combinations(cls, raw=False):
    #    """
    #    Generates reasonable parameter combinations.
    #
    #    Returns:
    #        list(dict): a list of meaningful parameter combinations
    #    """
    #    parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
    #                                             1.0, 1.5, 2.0],
    #                              'topology': ['star', 'bus',
    #                                           'poly_1', 'poly_2', 'poly_3']}
    #    return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # extracting minority samples
        X_min = X[y == self.min_label]

        samples = []
        if self.topology == 'star':
            # Implementation of the star topology
            kdx = np.max([1, int(np.rint(n_to_sample / X_min.shape[0]))])
            splits = np.arange(1, kdx + 1).astype(float)/(kdx + 1)

            X_mean = np.mean(X_min, axis=0) # pylint: disable=invalid-name
            diffs = X_mean - X_min

            samples = np.vstack(diffs[:, None] * splits[:, None])
            samples = samples + np.repeat(X_min,
                                            np.repeat(kdx, X_min.shape[0]),
                                            axis=0)
        elif self.topology == 'bus':
            # Implementation of the bus topology
            kdx = np.max([1, int(np.rint(n_to_sample / X_min.shape[0]))])
            splits = np.arange(1, kdx + 1).astype(float)/(kdx + 1)

            diffs = np.diff(X_min, axis=0)

            samples = np.vstack(diffs[:, None] * splits[:, None])

            samples = samples + np.repeat(X_min[:-1],
                                            np.repeat(kdx, X_min.shape[0] - 1),
                                            axis=0)
        elif self.topology.startswith('poly'):
            # Implementation of the polynomial topology
            degree = int(self.topology.split('_')[1])

            # this hack is added to make the fitted polynoms independent
            # from the ordering of the minority samples
            X_min = X_min[np.mean(X_min, axis=1).argsort()]

            def fit_poly(dim):
                return np.poly1d(np.polyfit(np.arange(len(X_min)),
                                            X_min[:, dim],
                                            degree))

            polys = [fit_poly(dim) for dim in range(X_min.shape[1])]

            for dim in range(X_min.shape[1]):
                rand = self.random_state.random_sample(size=n_to_sample)\
                                                                * X_min.shape[0]
                samples.append(np.array(polys[dim](rand)))

            samples = np.vstack(samples).T

        return (np.vstack([X, np.vstack(samples)]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'topology': self.topology,
                **OverSampling.get_params(self)}

class polynom_fit_SMOTE_star(OverSampling):# pylint: disable=invalid-name
    """
    References:
        * BibTex::

            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.polynom_fit_smote = polynom_fit_SMOTE(proportion=proportion,
                                                    topology='star',
                                                    random_state=random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        params = self.polynom_fit_smote.get_params()
        params ['class_name'] = self.__class__.__name__
        return params

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        return self.polynom_fit_smote.sample(X, y)

class polynom_fit_SMOTE_bus(OverSampling):# pylint: disable=invalid-name
    """
    References:
        * BibTex::

            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.polynom_fit_smote = polynom_fit_SMOTE(proportion=proportion,
                                                    topology='bus',
                                                    random_state=random_state)

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        params = self.polynom_fit_smote.get_params()
        params ['class_name'] = self.__class__.__name__
        return params

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        return self.polynom_fit_smote.sample(X, y)

class polynom_fit_SMOTE_poly(OverSampling):# pylint: disable=invalid-name
    """
    References:
        * BibTex::

            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 order=2,
                 *,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            order (int): the order of the fitted polynoms
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__(random_state=random_state)
        self.polynom_fit_smote = polynom_fit_SMOTE(proportion=proportion,
                                                    topology='poly_' + str(order),
                                                    random_state=random_state)
        self.order = order

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0],
                                  'order': [1, 2, 3]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        params = {'order': self.order,
                **self.polynom_fit_smote.get_params()}
        params['class_name'] = self.__class__.__name__
        return params

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        return self.polynom_fit_smote.sample(X, y)

class polynom_fit_SMOTE_mesh(OverSamplingSimplex): # pylint: disable=invalid-name
    """
    References:
        * BibTex::

            @INPROCEEDINGS{polynomial_fit_smote,
                            author={Gazzah, S. and Amara, N. E. B.},
                            booktitle={2008 The Eighth IAPR International
                                        Workshop on Document Analysis Systems},
                            title={New Oversampling Approaches Based on
                                    Polynomial Fitting for Imbalanced Data
                                    Sets},
                            year={2008},
                            volume={},
                            number={},
                            pages={677-684},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSamplingSimplex.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 ss_params=None,
                 *,
                 random_state=None,
                 **_kwargs):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            ss_params (dict): simplex sampling parameters
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        ss_params_default = {'n_dim': 2, 'simplex_sampling': 'uniform',
                            'within_simplex_sampling': 'deterministic',
                            'gaussian_component': None}
        ss_params = coalesce_dict(ss_params, ss_params_default)

        super().__init__(**ss_params, random_state=random_state)
        self.check_greater_or_equal(proportion, "proportion", 0.0)

        self.proportion = proportion

    @ classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
                                                 1.0, 1.5, 2.0]}
        return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion)

        if n_to_sample == 0:
            return self.return_copies(X, y, "Sampling is not needed")

        # extracting minority samples
        X_min = X[y == self.min_label]

        kdx = np.max([1, int(np.rint(n_to_sample / X_min.shape[0]))])

        samples = []

        base = np.arange(X_min.shape[0])
        neighbors = np.vstack([self.random_state.choice(np.arange(X_min.shape[0]),
                                        X_min.shape[0],
                                        replace=True) for _ in range(kdx)])

        indices = np.vstack([base, neighbors]).T

        samples = self.sample_simplex(X=X_min,
                                        indices=indices,
                                        n_to_sample=n_to_sample)


        ## Implementation of the mesh topology
        #if len(X_min)**2 > n_to_sample:
        #    while len(samples) < n_to_sample:
        #        random_i = self.random_state.randint(len(X_min))
        #        random_j = self.random_state.randint(len(X_min))
        #        diff = X_min[random_i] - X_min[random_j]
        #        samples.append(X_min[random_j] + 0.5*diff)
        #else:
        #    n_combs = (len(X_min)*(len(X_min)-1)/2)
        #    k = max([1, int(np.rint(n_to_sample/n_combs))])
        #    for i in range(len(X_min)):
        #        for j in range(len(X_min)):
        #            diff = X_min[i] - X_min[j]
        #            for li in range(1, k+1):
        #                samples.append(X_min[j] + float(li)/(k+1)*diff)

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                **OverSamplingSimplex.get_params(self)}
