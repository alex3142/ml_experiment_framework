from typing import (
    Dict,
    Any,
    List,
    Tuple,
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.dummy import (
    DummyClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier,
)

from modelling.modelling_utils.sk_pipeline_transformers import ColumnKeeper


def column_transformer_factory(
        column_transformer_config: Dict[str, Dict[str, Any]],
        column_transformer_kwargs: Dict[str, Dict[str, Any]],
) -> ColumnTransformer:
    """
    Get the actual pipeline requested
    :param column_transformer_config: Dict[str, Dict[str, Any]] - information about required columns transformers
    :param column_transformer_kwargs: Dict[str, Dict[str, Any]] - kwargs for column transformer
    :param transformer_map: Dict[str, Any], - all transformers available to ue for the model pipeline
    :return: Pipeline - model pipeline as requested
    """

    known_transformers = (
        SimpleImputer,
        OneHotEncoder,
        StandardScaler,
    )

    transformer_map = {trans.__name__: trans for trans in known_transformers}

    column_transformer_pipeline = []
    for current_trans_id, current_trans_info in column_transformer_config.items():
        column_transformer_pipeline.append(
            (
                current_trans_id,
                transformer_map[current_trans_info['trans_name']](
                    **current_trans_info['trans_args']
                ).set_output(transform='pandas'),
                current_trans_info['col_names'],
            )
        )

    return ColumnTransformer(
        column_transformer_pipeline,
        **column_transformer_kwargs,
    ).set_output(transform='pandas')


def model_factory(
        model_config: List[Dict[str, Any]],
) -> List[Tuple[Any, Any]]:
    """
    Get the actual pipeline requested
    :param column_trans: ColumnTransformer - column tranformations
    :param model_config: Dict[str, Dict[str, Any]] - ML model information
    :param transformer_map: Dict[str, Any], - all transformers available to ue for the model pipeline
    :return: Pipeline - model pipeline as requested
    """

    known_models = (
        DummyClassifier,
        LogisticRegression,
        RandomForestClassifier,
        StandardScaler,
        ColumnKeeper,
    )

    models_map = {model.__name__: model for model in known_models}

    model_pipeline = []
    for current_model in model_config:
        if hasattr(models_map[current_model['type']], 'set_output'):
            model_pipeline.append(
                (
                    current_model['type'],
                    models_map[current_model['type']](
                        **current_model['kwargs']
                    ).set_output(transform='pandas')
                )
            )
        else:
            model_pipeline.append(
                (
                    current_model['type'],
                    models_map[current_model['type']](
                        **current_model['kwargs']
                    )
                )
            )

    return model_pipeline


def pipeline_factory(
        exp_config: Dict[str, Any],
) -> Pipeline:
    """
    Controls the creation of pipelines
    :param exp_config: Dict[str, Dict[str, Any]] - experiment information
    :param transformer_map: Dict[str, Any], - all transformers available to ue for the model pipeline
    :return: Pipeline - model pipeline as requested
    """

    col_trans = column_transformer_factory(
        column_transformer_config=exp_config['column_transformer_map'],
        column_transformer_kwargs=exp_config['column_transformer_args'],
    ).set_output(transform='pandas')
    model_pipeline = model_factory(
        model_config=exp_config['model_pipline'],
    )

    full_pipeline = [('column_trans', col_trans)]
    full_pipeline.extend(model_pipeline)
    return Pipeline(full_pipeline).set_output(transform='pandas')

