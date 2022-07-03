from ._OverSampling import OverSampling

from .._logger import logger
_logger= logger

__all__= ['NoSMOTE']

class NoSMOTE(OverSampling):
    """
    The goal of this class is to provide a functionality to send data through
    on any model selection/evaluation pipeline with no oversampling carried
    out. It can be used to get baseline estimates on preformance.
    """

    categories = []

    def __init__(self, nn_params=None, random_state=None, n_jobs=None):
        """
        Constructor of the NoSMOTE object.

        Args:
            random_state (int/np.random.RandomState/None): dummy parameter for \
                        the compatibility of interfaces
        """
        super().__init__()

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            list(dict): a list of meaningful parameter combinations
        """
        return cls.generate_parameter_combinations({}, raw=False)

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

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {}
