from typing import (
    Dict,
    Any,
)

import pandas as pd


def calculate_clf_weight(y_true: int, exposure: float) -> float:
    """
    this one is a bit weird, if you have full exposure then weight is 1
    if you have less than full exposure and 1 label,
    then we know it's a 1 for certain regardless of
    exposure so make weight 1
    if you have zero but non full exposure weight according to expsoure,

    This is needed to squeeze a Poisson shaped model into a Bernoulli
    shaped hole.

    :param y_true:
    :param exposure:
    :return:
    """

    if y_true == 1:
        return 1
    if exposure >= 1:
        return 1
    return exposure


def binarise_labels(
        input_data: pd.DataFrame,
        exp_config: Dict[str, Any],
    ) -> pd.DataFrame:
    """
    this one is a bit weird, if you have full exposure then weight is 1
    if you have less than full exposure and 1 label,
    then we know it's a 1 for certain regardless of
    exposure so make weight 1
    if you have zero but non full exposure weight according to expsoure,

    This is needed to squeeze a Poisson shaped model into a Bernoulli
    shaped hole.

    :param y_true:
    :param exposure:
    :return:
    """

    original_labels = input_data[exp_config['label_col_name']]
    input_data = input_data.drop(columns=[exp_config['label_col_name']])
    binary_labels = (original_labels != 0).astype(int)

    input_data = input_data.assign(
        **{exp_config['label_col_name']: binary_labels}
    )

    binary_exposure = input_data.apply(lambda row: calculate_clf_weight(
            y_true=row[exp_config['label_col_name']],
            exposure=row[exp_config['sample_weight_col_name']]
    ),
    axis=1
    )
    input_data = input_data.drop(columns=[exp_config['sample_weight_col_name']])

    input_data = input_data.assign(
        **{exp_config['sample_weight_col_name']: binary_exposure}
    )

    return input_data