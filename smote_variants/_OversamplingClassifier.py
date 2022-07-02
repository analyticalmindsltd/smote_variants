from sklearn.base import BaseEstimator, ClassifierMixin

class OversamplingClassifier(BaseEstimator, ClassifierMixin):
    """
    This class wraps an oversampler and a classifier, making it compatible
    with sklearn based pipelines.
    """

    def __init__(self, oversampler, classifier):
        """
        Constructor of the wrapper.

        Args:
            oversampler (obj): an oversampler object
            classifier (obj): an sklearn-compatible classifier
        """

        self.oversampler = oversampler
        self.classifier = classifier

    def fit(self, X, y=None):
        """
        Carries out oversampling and fits the classifier.

        Args:
            X (np.ndarray): feature vectors
            y (np.array): target values

        Returns:
            obj: the object itself
        """

        X_samp, y_samp = self.oversampler.sample(X, y)
        self.classifier.fit(X_samp, y_samp)

        return self

    def predict(self, X):
        """
        Carries out the predictions.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Carries out the predictions with probability estimations.

        Args:
            X (np.ndarray): feature vectors
        """

        return self.classifier.predict_proba(X)

    def get_params(self, deep=True):
        """
        Returns the dictionary of parameters.

        Args:
            deep (bool): wether to return parameters with deep discovery

        Returns:
            dict: the dictionary of parameters
        """

        return {'oversampler': self.oversampler, 'classifier': self.classifier}

    def set_params(self, **parameters):
        """
        Sets the parameters.

        Args:
            parameters (dict): the parameters to set.

        Returns:
            obj: the object itself
        """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self


