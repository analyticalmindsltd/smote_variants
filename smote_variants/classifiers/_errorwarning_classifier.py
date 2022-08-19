"""
A dummy classifier raising errors and warnings.
"""

import warnings

from sklearn.neighbors import KNeighborsClassifier

__all__=['ErrorWarningClassifier']

class ErrorWarningClassifier:
    """
    Mock classifier throwing an error or warning.
    """

    def __init__(self, raise_value_error=False,
                       raise_warning=False,
                       raise_runtime_error=False):
        """
        Constructor of the classifier.

        Args:
            raise_value_error (bool): whether to raise a ValueError
            raise_warning (bool): whether to raise a warning
            raise_runtime_error (bool): whether to raise a RuntimeError
        """
        self.raise_value_error = raise_value_error
        self.raise_warning = raise_warning
        self.raise_runtime_error = raise_runtime_error
        self.classifier = KNeighborsClassifier()

    def fit(self, X, y):
        """
        Fitting the classifier

        Args:
            X (np.array): the training vectors
            y (np.array): the target labels

        Returns:
            obj: the fitted object
        """

        if self.raise_value_error:
            raise ValueError("Dummy value error")

        if self.raise_runtime_error:
            raise RuntimeError("Dummy runtime error")

        if self.raise_warning:
            warnings.warn("Dummy warning")

        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the new vectors.

        Args:
            X (np.array): new vectors to predict

        Returns:
            np.array: the predicted labels
        """

    def predict_proba(self, X):
        """
        Predict probabilities for the new vectors.

        Args:
            X (np.array): new vectors to predict

        Returns:
            np.array: the predicted probabilities
        """
        return self.classifier.predict_proba(X)

    def get_params(self):
        """
        Return the parameters

        Returns:
            dict: the parameters
        """
        return {'raise_value_error': self.raise_value_error,
                'raise_warning': self.raise_warning,
                'raise_runtime_error': self.raise_runtime_error}
