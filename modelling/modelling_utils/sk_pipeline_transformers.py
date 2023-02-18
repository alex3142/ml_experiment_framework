from typing import List

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)


class ColumnKeeper(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            col_names_to_keep: List[str]
    ) -> None:
        self.col_names_to_keep = col_names_to_keep

    def get_feature_names_out(self, input_features=None):
        """
        Needed to use pandas output
        :param input_features:
        :return:
        """
        return self.col_names_to_keep

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.col_names_to_keep]

    def fit(self, X, y=None, **fit_params):
        return self
