import pandas as pd
import yaml
from typing import (
    Dict,
    Any,
    Tuple,
    Callable,
    Optional,
)
from pathlib import Path

import numpy as np
import mlflow
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import (
    GridSearchCV,
    KFold
)
from sklearn.metrics import (
    make_scorer,
    r2_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline
import click

from modelling.model_bases import (
    DummyRegressorRounder,
    RandomForestRegressorRounder,
    LinearRegressionRounder,
    Rounder,
)

known_transformers = (
    DummyRegressorRounder,
    RandomForestRegressorRounder,
    LinearRegressionRounder,
    TfidfVectorizer,
    CountVectorizer,
    TruncatedSVD,
    Rounder,
)


def parse_config(experiment_dir: Optional[str], config_file: str) -> Dict[str, Any]:
    """
    Parse the configuration file for the experiment
    :param experiment_dir:  Optional[str] - directory of experiment configuration files
    :param config_file: str - configuration filename
    :return: Dict[str, Any] loaded configuration file for experiment
    """

    if experiment_dir is None:
        experiment_dir = Path(__file__).parent.joinpath('experiment_configs')
    else:
        experiment_dir = Path(experiment_dir)

    config_file_full = experiment_dir.joinpath(config_file)
    config = yaml.load(open(config_file_full).read(), Loader=yaml.FullLoader)
    config['config_filename'] = str(config_file_full)
    print('config loaded.')

    return config


def pipeline_factory(
        pipeline_config: Dict[str, Dict[str, Any]],
        known_transformers: Tuple[Callable, ...],
) -> Pipeline:
    """
    Get the actual pipeline requested
    :param pipeline_config: Dict[str, Dict[str, Any]] - pipeline configuration
    :param known_transformers: Tuple[Callable, ...] - all transformers available to ue for the model pipeline
    :return: Pipeline - model pipeline as requested
    """

    transformer_map = {trans.__name__: trans for trans in known_transformers}

    model_pipeline = []
    for trans_name, trans_kwargs in pipeline_config.items():
        model_pipeline.append((trans_name, transformer_map[trans_name](**trans_kwargs)))

    return Pipeline(model_pipeline)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load data from filename
    :param filename: str - filename to load data from
    :return: pd.DataFrame - loaded data
    """

    data = pd.read_csv(filename).fillna('')
    print('data loaded.')
    return data


def test_model(
        pipeline: Pipeline,
        experiment_config: Dict[str, Any],
) -> None:
    """
    Runs the final model on the test set
    :param pipeline: Pipeline - fitted model pipeline
    :param experiment_config: Dict[str, Any] - experiment configuration data
    :return:
    """

    test_data = load_data(filename=experiment_config['test_data'])
    test_set_results = pipeline.predict(test_data[experiment_config['features_col_name']])
    mlflow.log_metric(
        'test_balanced_accuracy', balanced_accuracy_score(
            y_true=test_data[experiment_config['label_col_name']], y_pred=test_set_results)
    )
    mlflow.log_metric(
        'test_r2', r2_score(
            y_true=test_data[experiment_config['label_col_name']], y_pred=test_set_results)
    )


def train_model(
        train_data: pd.DataFrame,
        experiment_config: Dict[str, Any],
        known_transformers: Tuple[Callable, ...],
) -> None:
    """
    trains the model, performs hyperparameter search, produces cross validation metrics,
    trains and stores final model
    :param train_data: pd.DataFrame - training data
    :param experiment_config: Dict[str, Any] - experiment configuration data
    :param known_transformers: Tuple[Callable, ...] - all transformers available to ue for the model pipeline
    :return:
    """

    print('training model...')
    with mlflow.start_run(run_name=experiment_config['config_filename']) as run:

        mlflow.log_param('experiment_config', experiment_config['config_filename'])
        mlflow.log_param('data_shape', str(train_data.shape))
        mlflow.log_param('data_dtypes', str(train_data.dtypes))

        cv = KFold(experiment_config['n_folds'])
        best_params = {}

        if experiment_config.get('param_grid') is not None:
            pipeline = pipeline_factory(
                pipeline_config=experiment_config['pipeline_map'], known_transformers=known_transformers
            )
            cv = GridSearchCV(
                estimator=pipeline,
                param_grid=experiment_config.get('param_grid'),
                scoring=make_scorer(r2_score),
                refit=False,
                cv=cv
            )
            cv.fit(X=train_data[experiment_config['features_col_name']], y=train_data[experiment_config['label_col_name']])
            best_params = cv.best_params_

            mlflow.log_param('best_params', str(cv.best_params_))

        balanced_accuracy_results = []
        r2_results = []
        for train_index, val_index in cv.split(train_data):
            pipeline = pipeline_factory(
                pipeline_config=experiment_config['pipeline_map'], known_transformers=known_transformers
            )
            pipeline.set_params(**best_params)
            train_cv = train_data.iloc[train_index]
            val_cv = train_data.iloc[val_index]
            pipeline.fit(
                X=train_cv[experiment_config['features_col_name']], y=train_cv[experiment_config['label_col_name']]
            )

            pred_cv = pipeline.predict(val_cv[experiment_config['features_col_name']])

            balanced_accuracy_results.append(balanced_accuracy_score(
                y_true=val_cv[experiment_config['label_col_name']],
                y_pred=pred_cv)
            )

            r2_results.append(r2_score(
                y_true=val_cv[experiment_config['label_col_name']],
                y_pred=pred_cv)
            )

        [mlflow.log_metric('r2', v, step=step) for v, step in enumerate(r2_results)]
        mlflow.log_metric('r2_mean', np.array(r2_results).mean())

        [mlflow.log_metric('balanced_accuracy', v, step=step) for v, step in enumerate(balanced_accuracy_results)]
        mlflow.log_metric('balanced_accuracy_mean', np.array(balanced_accuracy_results).mean())

        if 'test_data' in experiment_config:
            pipeline = pipeline_factory(
                pipeline_config=experiment_config['pipeline_map'], known_transformers=known_transformers
            )
            pipeline.set_params(**best_params)
            pipeline.fit(
                X=train_data[experiment_config['features_col_name']], y=train_data[experiment_config['label_col_name']]
            )
            test_model(pipeline=pipeline, experiment_config=experiment_config)

            config_filename = Path(experiment_config['config_filename']).stem
            mlflow.sklearn.save_model(
                pipeline,
                path=str(
                    Path(__file__).parent.joinpath('models').joinpath(config_filename)
                )
            )


@click.command()
@click.option('--experiment_dir', default=None, help='Directory of experiment config.')
@click.option('--config_file',  help='configuration file name.')
def main(experiment_dir: Optional[str], config_file: str) -> None:
    """
    Main function and entry point to processing
    :param experiment_dir: Optional[str] - directory of configuration file
    :param config_file: str - configuration file name
    :return:
    """

    config = parse_config(experiment_dir=experiment_dir, config_file=config_file)
    train_data = load_data(filename=config['train_data'])
    train_model(train_data=train_data, experiment_config=config, known_transformers=known_transformers)

    print('experiment complete, have a nice day')


if __name__ == "__main__":
    main()
