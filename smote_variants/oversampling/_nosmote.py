"""
This module implements the NoSMOTE method.
"""

from ..base import OverSampling

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

    def __init__(self, raise_value_error=False,
                        raise_runtime_error=False,
                        random_state=None,
                        **_kwargs):
        """
        Constructor of the NoSMOTE object.

        Args:
            random_state (int/np.random.RandomState/None): dummy parameter for \
                        the compatibility of interfaces
            raise_value_error (bool): whether to raise a ValueError
            raise_runtime_error (bool): whether to raise a RuntimeError
        """
        super().__init__(random_state=random_state)

        self.raise_value_error = raise_value_error
        self.raise_runtime_error = raise_runtime_error

    @classmethod
    def parameter_combinations(cls, raw=False):
        """
        Generates reasonable parameter combinations.

        Returns:
            raw (bool): whether to return the raw parameter generation setup
        """
        return cls.generate_parameter_combinations({}, raw=raw)

    def sampling_algorithm(self, X, y):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """

        if self.raise_value_error:
            raise ValueError("Dummy ValueError raised by NoSMOTE")

        if self.raise_runtime_error:
            raise RuntimeError("Dummy RuntimeError raised by NoSMOTE")

        return X.copy(), y.copy()

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'raise_value_error': self.raise_value_error,
                'raise_runtime_error': self.raise_runtime_error,
                **OverSampling.get_params(self)}
