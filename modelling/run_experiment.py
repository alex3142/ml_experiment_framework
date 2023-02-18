import pickle
import logging
from pathlib import Path

import pandas as pd
from typing import (
    Dict,
    Any,
    Callable,
    Optional,
)
import hashlib

import numpy as np
import mlflow
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    TimeSeriesSplit
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    make_scorer,
)
import click
import matplotlib.pyplot as plt

from modelling.modelling_utils.metrics import prediction_prop_score
from modelling.modelling_utils.labels_and_weights_processing import (
    binarise_labels,
)
from modelling.modelling_utils.model_and_transformer_factories import (
    column_transformer_factory,
    pipeline_factory,
    model_factory,
)
from modelling.modelling_utils.loaders import (
    load_data,
    parse_config,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

KNOWN_METRICS = (
    roc_auc_score,
    precision_score,
    recall_score,
    prediction_prop_score,
)


def feature_selection(
        exp_config: Dict[str, Any],
        train_data: pd.DataFrame,
) -> None:
    """
    TODO - this whole function needs work, investigate better structuring
    and solutions to feature extraction

    :param exp_config:
    :param train_data:
    :return:
    """

    logger.info('running feature selection')

    # Piplines seem to have some issues with feature selection so manually
    # separating - more work could be done to investigate and improve this

    col_trans = column_transformer_factory(
        column_transformer_config=exp_config['column_transformer_map'],
        column_transformer_kwargs=exp_config['column_transformer_args'],
    )

    col_trans_pipe = [('column_trans', col_trans)]

    if exp_config['model_pipline'][0]['type'] == 'StandardScaler':
        col_trans_pipe.append(
            ('standard_scaler', StandardScaler().set_output(transform='pandas')
             )
        )

    col_trans_pipe = Pipeline(col_trans_pipe).set_output(transform='pandas')

    # run feature selection
    mod = model_factory(
        model_config=exp_config['model_pipline']
    )[-1][-1]
    selector = RFE(mod, step=1, n_features_to_select=1)

    trans_data = col_trans_pipe.fit_transform(
        X=train_data[exp_config['features_col_names']],
        y=train_data[exp_config['label_col_name']],
    )

    selector = selector.fit(
        X=trans_data,
        y=train_data[exp_config['label_col_name']],
    )

    results = []
    VAL_SET_SIZE = 2500

    for i in range(selector.ranking_.max()):
        mask = selector.ranking_ <= i + 1

        mod = model_factory(
            model_config=exp_config['model_pipline']
        )[-1][-1]
        trans_data_masked = trans_data.iloc[:, mask]

        mod.fit(
            X=trans_data_masked.iloc[:-VAL_SET_SIZE],
            y=train_data[exp_config['label_col_name']].iloc[:-VAL_SET_SIZE],
            sample_weight=train_data[
                              exp_config['sample_weight_col_name']
                          ].iloc[:-VAL_SET_SIZE]
        )

        results.append(roc_auc_score(
            y_true=train_data[exp_config['label_col_name']].iloc[
                   -VAL_SET_SIZE:],
            y_score=mod.predict(trans_data_masked.iloc[-VAL_SET_SIZE:]),
            sample_weight=train_data[
                              exp_config['sample_weight_col_name']
                          ].iloc[-VAL_SET_SIZE:],
        ))

    fig = plt.figure()
    plt.scatter(x=range(len(results)), y=results)
    mlflow.log_figure(fig, "feature_selection.png")

    pd.DataFrame(
        zip(col_trans_pipe.get_feature_names_out(), selector.ranking_),
        columns=['name', 'rank']).sort_values(by=['rank']
                                              ).to_html("feature_ranks.html")
    mlflow.log_artifact("feature_ranks.html",
                        "feature_ranks")


def model_understanding(
        model_pipeline: Pipeline
) -> None:
    """

    This need refactoring, lots of unobvious diggigng around pipelines
    which may break in future

    :return:
    """
    logger.info('running model understanding')
    feature_values = None
    columns = None
    if hasattr(model_pipeline.steps[-1][1], 'coef_'):
        feature_values = list(
            zip(
                model_pipeline.steps[-2][1].get_feature_names_out(),
                model_pipeline.steps[-1][1].coef_[0]
            )
        )

        feature_values.append(
            ('intercept', model_pipeline.steps[-1][1].intercept_[0]))
        columns = ['names', 'importance']

    elif hasattr(model_pipeline.steps[-1][1], 'feature_importances_'):
        feature_values = zip(
            model_pipeline.steps[-2][1].get_feature_names_out(),
            model_pipeline.steps[-1][1].feature_importances_,
        )
        columns = ['names', 'importance']

    if feature_values is not None:
        pd.DataFrame(
            feature_values, columns=columns
        ).sort_values(
            by=['importance'], ascending=False
        ).to_html('feature_importance.html')
        mlflow.log_artifact("feature_importance.html",
                            "feature_importance")


def tune_hyperparams(
        model: Callable,
        cv: Callable,
        exp_config: Dict[str, Any],
        train_data: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Runs hp tuning (only grid search at the moment)
    :param model: Callable - model to tue
    :param exp_config: Dict[str, Any] loaded configuration file for experiment
    :param train_data: pd.DataFrame
    :return: Dict[str, Any] - best parameters
    """

    logger.info('running hp tuning')

    sample_weight_work_around = {
        f'{model.steps[-1][0]}__sample_weight': train_data[
            exp_config['sample_weight_col_name']]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=exp_config.get('param_grid'),
        refit=False,
        cv=cv,
    )
    grid_search.fit(
        X=train_data[exp_config['features_col_names']],
        y=train_data[exp_config['label_col_name']],
        **sample_weight_work_around
    )

    mlflow.log_param('best_params', str(grid_search.best_params_))

    return grid_search.best_params_


def log_metrics(
        metrics_store: Dict[str, np.ndarray]) -> None:
    """
    Logs the metrics to mlflow
    :param metric_store:
    :return:
    """

    for met_name, met_results in metrics_store.items():
        [
            mlflow.log_metric(met_name, v, step=step) for step, v in
            enumerate(met_results)
        ]

        if len(met_results) > 1:
            mlflow.log_metric(
                f'{met_name}_mean', met_results.mean())


def run_cv(
        cv: Callable,
        train_data: pd.DataFrame,
        exp_config: Dict[str, Any],
        best_params: Dict[str, Any],
) -> None:
    """
    https://github.com/scikit-learn/scikit-learn/issues/18159 - annoying
    :param cv:
    :param train_data:
    :param models_map:
    :param transformers_map:
    :param exp_config:
    :param best_params:
    :return:
    """

    logger.info('running cv')

    mod = pipeline_factory(
        exp_config=exp_config,
    )

    mod.set_params(**best_params)

    sample_weight_work_around = {
        f'{mod.steps[-1][0]}__sample_weight': train_data[
            exp_config['sample_weight_col_name']]
    }

    cv_results = cross_validate(
        estimator=mod,
        X=train_data[exp_config['features_col_names']],
        y=train_data[exp_config['label_col_name']],
        fit_params=sample_weight_work_around,
        cv=cv,
        scoring={
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score),
            'roc_auc': make_scorer(roc_auc_score),
            'prediction_prop_score': make_scorer(prediction_prop_score),
        }
    )

    # it automatically adds 'test' to all metrics which is confusing with
    # test set metrics
    cv_results_clean = {
        k.replace('test_', ''): v for k, v in cv_results.items()
    }

    log_metrics(metrics_store=cv_results_clean)


def train_final_model(
        train_data: pd.DataFrame,
        exp_config: Dict[str, Any],
        best_params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    logger.info('training final model')

    mod = pipeline_factory(
        exp_config=exp_config,
    )

    mod.set_params(**best_params)

    sample_weight_work_around_train = {
        f'{mod.steps[-1][0]}__sample_weight': train_data[
            exp_config['sample_weight_col_name']]
    }

    mod.fit(
        X=train_data[exp_config['features_col_names']],
        y=train_data[exp_config['label_col_name']],
        **sample_weight_work_around_train
    )

    return mod


def evaluate_final_model(
        mod: Pipeline,
        test_data: pd.DataFrame,
        exp_config: Dict[str, Any],
) -> None:
    metrics_to_run = (
        recall_score,
        precision_score,
        roc_auc_score,
        prediction_prop_score,
    )

    metrics_to_run = {met.__name__: met for met in metrics_to_run}

    metric_results = {}
    for metric_name, metric_func in metrics_to_run.items():
        # can't use key word arguments as scorer kwargs differ
        # wrapping in numpy as cv results are numpy
        metric_results[f'test__{metric_name}'] = np.array([
            metric_func(
                test_data[exp_config['label_col_name']],
                mod.predict(
                    test_data[exp_config['features_col_names']],
                ),
                sample_weight=test_data[exp_config['sample_weight_col_name']],
            )
        ])

    log_metrics(metrics_store=metric_results)


def save_model(
        mod: Pipeline,
        exp_config: Dict[str, Any],
) -> None:
    model_name = Path(__file__).parent.joinpath(
        exp_config['model_save_path']
    ).joinpath(exp_config['model_name'])
    with open(model_name, 'wb') as f:
        pickle.dump(mod, f)

    logger.info('model saved')


def run_experiment(
        train_data: pd.DataFrame,
        exp_config: Dict[str, Any],
) -> None:
    """
    trains the model, performs hyperparameter search, produces cross validation metrics,
    trains and stores final model
    :param train_data: pd.DataFrame - training data
    :param exp_config: Dict[str, Any] - experiment configuration data
    :param known_metrics: Tuple[Callable, ...] - all transformers available to ue for the model pipeline
    :return:
    """

    logger.info('running experiment')

    if exp_config['binarise_label']:
        train_data = binarise_labels(
            input_data=train_data, exp_config=exp_config
        )

    with mlflow.start_run(run_name=exp_config['config_filename']) as run:

        mlflow.log_param(
            'train_data_hash',
            hashlib.sha1(
                pd.util.hash_pandas_object(train_data).values
            ).hexdigest()
        )

        mlflow.log_param('exp_config', exp_config['config_filename'])
        mlflow.log_param('data_shape', str(train_data.shape))
        #
        cv = TimeSeriesSplit(n_splits=exp_config['n_folds'])
        best_params = exp_config.get('best_params')

        ############################################################
        #                 Run feature selection                    #
        ############################################################

        if exp_config['run_feature_selection']:
            feature_selection(
                exp_config=exp_config,
                train_data=train_data,
            )

        ############################################################
        #                   Run HP tuning                          #
        ############################################################
        if (
                len(best_params) == 0
        ) \
                and (
                exp_config.get('param_grid') is not None
        ):
            mod = pipeline_factory(
                exp_config=exp_config,
            )

            best_params = tune_hyperparams(
                model=mod,
                cv=cv,
                exp_config=exp_config,
                train_data=train_data,
            )

        ############################################################
        #                          Run CV                          #
        ############################################################
        run_cv(
            cv=cv,
            train_data=train_data,
            exp_config=exp_config,
            best_params=best_params,
        )

        ###############################################################
        #                     train final model                       #
        ###############################################################
        final_mod = train_final_model(
            train_data=train_data,
            exp_config=exp_config,
            best_params=best_params,
        )

        ###############################################################
        #                  understand final model                     #
        ###############################################################
        model_understanding(model_pipeline=final_mod)

        ###############################################################
        #                  Evaluate final model                       #
        ###############################################################

        if exp_config['evaluate_final_model']:

            test_data = load_data(
                filename=exp_config['test_data'],
                load_data_kwargs=exp_config['data_load_kwargs'],
                file_path=exp_config['data_file_path']
            )

            if exp_config['binarise_label']:
                test_data = binarise_labels(
                    input_data=test_data, exp_config=exp_config
                )

            evaluate_final_model(
                mod=final_mod,
                test_data=test_data,
                exp_config=exp_config),

        ###############################################################
        #                       save final model                      #
        ###############################################################
        if exp_config.get('save_model', False):
            save_model(mod=final_mod, exp_config=exp_config)


@click.command()
@click.option('--experiment_dir', default=None,
              help='Directory of experiment config.')
@click.option('--config_file', help='configuration file name.')
def main(experiment_dir: Optional[str], config_file: str) -> None:
    """
    Main function and entry point to processing
    :param experiment_dir: Optional[str] - directory of configuration file
    :param config_file: str - configuration file name
    :return:
    """

    config = parse_config(experiment_dir=experiment_dir,
                          config_file=config_file)
    train_data = load_data(
        filename=config['train_data'],
        load_data_kwargs=config['data_load_kwargs'],
        file_path=config['data_file_path']
    )
    run_experiment(
        train_data=train_data,
        exp_config=config,
    )

    logger.info('experiment complete, have a nice day')


if __name__ == "__main__":
    main()
