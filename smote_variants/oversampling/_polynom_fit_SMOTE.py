import numpy as np

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['polynom_fit_SMOTE']

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
                            keywords={curve fitting;learning (artificial
                                        intelligence);mesh generation;pattern
                                        classification;polynomials;sampling
                                        methods;support vector machines;
                                        oversampling approach;polynomial
                                        fitting function;imbalanced data
                                        set;pattern classification task;
                                        class-modular strategy;support
                                        vector machine;true negative rate;
                                        true positive rate;star topology;
                                        bus topology;polynomial curve
                                        topology;mesh topology;Polynomials;
                                        Topology;Support vector machines;
                                        Support vector machine classification;
                                        Pattern classification;Performance
                                        evaluation;Training data;Text
                                        analysis;Data engineering;Convergence;
                                        writer identification system;majority
                                        class;minority class;imbalanced data
                                        sets;polynomial fitting functions;
                                        class-modular strategy},
                            doi={10.1109/DAS.2008.74},
                            ISSN={},
                            month={Sept},}
    """

    categories = [OverSampling.cat_extensive]

    def __init__(self,
                 proportion=1.0,
                 *,
                 topology='star',
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal
                                to the number of majority samples
            topoplogy (str): 'star'/'bus'/'mesh'
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0.0)
        if topology.startswith('poly'):
            self.check_greater_or_equal(
                int(topology.split('_')[-1]), 'topology', 1)
        else:
            self.check_isin(topology, "topology", ['star', 'bus', 'mesh'])

        self.proportion = proportion
        self.topology = topology

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
                                  'topology': ['star', 'bus', 'mesh',
                                               'poly_1', 'poly_2', 'poly_3']}
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

        # extracting minority samples
        X_min = X[y == self.min_label]

        # determine the number of samples to generate
        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        samples = []
        if self.topology == 'star':
            # Implementation of the star topology
            X_mean = np.mean(X_min, axis=0)
            k = max([1, int(np.rint(n_to_sample/len(X_min)))])
            for x in X_min:
                diff = X_mean - x
                for i in range(1, k+1):
                    samples.append(x + float(i)/(k+1)*diff)
        elif self.topology == 'bus':
            # Implementation of the bus topology
            k = max([1, int(np.rint(n_to_sample/len(X_min)))])
            for i in range(1, len(X_min)):
                diff = X_min[i-1] - X_min[i]
                for j in range(1, k+1):
                    samples.append(X_min[i] + float(j)/(k+1)*diff)
        elif self.topology == 'mesh':
            # Implementation of the mesh topology
            if len(X_min)**2 > n_to_sample:
                while len(samples) < n_to_sample:
                    random_i = self.random_state.randint(len(X_min))
                    random_j = self.random_state.randint(len(X_min))
                    diff = X_min[random_i] - X_min[random_j]
                    samples.append(X_min[random_i] + 0.5*diff)
            else:
                n_combs = (len(X_min)*(len(X_min)-1)/2)
                k = max([1, int(np.rint(n_to_sample/n_combs))])
                for i in range(len(X_min)):
                    for j in range(len(X_min)):
                        diff = X_min[i] - X_min[j]
                        for li in range(1, k+1):
                            samples.append(X_min[j] + float(li)/(k+1)*diff)
        elif self.topology.startswith('poly'):
            # Implementation of the polynomial topology
            deg = int(self.topology.split('_')[1])
            dim = len(X_min[0])

            def fit_poly(d):
                return np.poly1d(np.polyfit(np.arange(len(X_min)),
                                            X_min[:, d], deg))

            polys = [fit_poly(d) for d in range(dim)]

            for d in range(dim):
                random_sample = self.random_state.random_sample()*len(X_min)
                samples_gen = [polys[d](random_sample)
                               for _ in range(n_to_sample)]
                samples.append(np.array(samples_gen))
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
                'random_state': self._random_state_init}
