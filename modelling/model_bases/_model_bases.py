from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
