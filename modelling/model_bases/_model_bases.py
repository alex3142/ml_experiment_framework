from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
import numpy as np


class DummyRegressorRounder(DummyRegressor):
    """
    Wrapper to allow rounder to be used at end of pipeline
    """

    def transform(self, X):
        return self.predict(X)


class RandomForestRegressorRounder(RandomForestRegressor):
    """
    Wrapper to allow rounder to be used at end of pipeline
    """
    def transform(self, X):
        return self.predict(X)


class LinearRegressionRounder(LinearRegression):
    """
    Wrapper to allow rounder to be used at end of pipeline
    """
    def transform(self, X):
        return self.predict(X)


class Rounder:
    """
    This class rounds the final output of the pipeline to integer values between 1 and 5
    since the reviews are between integers between 1 and 5
    """
    def fit(self, *args, **kwargs):
        return self

    def fit_transform(self, *args, **kwargs):
        return self

    def predict(self, X: np.ndarray):
        X = np.where(X > 5, 5, X)
        X = np.where(X < 1, 1, X)
        return X.round()


class PoissonClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper to allow rounder to be used at end of pipeline
    """

    def __init__(self, model: Callable, decision_threshold: float = 0.5) -> None:

        self._model = model
        self._decision_threshold = decision_threshold

    def predict_proba(self, X) -> np.ndarray:
        return 1 - poisson.pmf(np.zeros(X.shape[0]), mu=self._model.predict(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self._decision_threshold).astype(int)

    def score(self, X, y, sample_weight=None):
        return balanced_accuracy_score(y_pred=self.predict(X), y_true=y, sample_weight=sample_weight)


