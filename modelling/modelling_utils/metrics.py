import numpy as np


def prediction_prop_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    **kwargs,
) -> float:
    """
    Calculates the proportion of the data which is given a positive prediction ",

    :param y_true:  np.ndarray not used but required by sklearn to be a scorer\n",
    :param y_pred: np.ndarray predictions from model\n",
    :return: float proportion of positive class predictions\n",
    """
    return y_pred.mean()