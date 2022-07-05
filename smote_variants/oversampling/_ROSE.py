import numpy as np

from ._OverSampling import OverSampling
from ._SMOTE import SMOTE

from .._logger import logger
_logger= logger

__all__= ['ROSE']

class ROSE(OverSampling):
    """
    References:
        * BibTex::

            @Article{rose,
                    author="Menardi, Giovanna
                    and Torelli, Nicola",
                    title="Training and assessing classification rules with
                            imbalanced data",
                    journal="Data Mining and Knowledge Discovery",
                    year="2014",
                    month="Jan",
                    day="01",
                    volume="28",
                    number="1",
                    pages="92--122",
                    issn="1573-756X",
                    doi="10.1007/s10618-012-0295-5",
                    url="https://doi.org/10.1007/s10618-012-0295-5"
                    }

    Notes:
        * It is not entirely clear if the authors propose kernel density
            estimation or the fitting of simple multivariate Gaussians
            on the minority samples. The latter seems to be more likely,
            I implement that approach.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_sample_componentwise]

    def __init__(self, 
                 proportion=1.0, 
                 *,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, 'proportion', 0.0)

        self.proportion = proportion

        self.set_random_state(random_state)

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

        n_to_sample = self.det_n_to_sample(self.proportion,
                                           self.class_stats[self.maj_label],
                                           self.class_stats[self.min_label])

        if n_to_sample == 0:
            _logger.warning(self.__class__.__name__ +
                            ": " + "Sampling is not needed")
            return X.copy(), y.copy()

        X_min = X[y == self.min_label]

        # Estimating the H matrix
        std = np.std(X_min, axis=0)
        d = len(X[0])
        n = len(X_min)
        H = std*(4.0/((d + 1)*n))**(1.0/(d + 4))

        # do the sampling
        samples = []
        for _ in range(n_to_sample):
            random_idx = self.random_state.randint(len(X_min))
            samples.append(self.sample_by_gaussian_jittering(
                X_min[random_idx], H))

        return (np.vstack([X, samples]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'random_state': self._random_state_init}
