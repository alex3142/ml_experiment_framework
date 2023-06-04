import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import pandas


log = logging.getLogger(__name__)


def metric_1(x, y):
    return 'metric_1 success'


def metric_2(features):
    return features.mean()


@hydra.main(
    config_path="../conf",
    config_name='config',
    version_base=None
)
def main(cfg: DictConfig) -> None:
    # get the hydra config to find the output directory for saving plots
    hydra_cfg = HydraConfig.get()

    # get the experiment name for the
    hydra.output_subdir = cfg.get('experiment_name')

    # you can use the config like a dict
    log.info(cfg.get("ergerger", []))

    # don't need to log the config as this happens for free in hydra
    # log.info(cfg)

    # get an example of pd data
    test_df = pd.DataFrame({'col_1': [1, 2], 'col_2': [20, 0.1]})

    # log the data frame
    log.info('\n\t' + test_df.to_string().replace('\n', '\n\t'))

    # create a figure to save
    fig = test_df['col_1'].plot(kind='bar',
                            figsize=(20, 16), fontsize=26).get_figure()

    some_variable_1 = 'hello'
    some_variable_2 = 'world'

    # save the figure using the hydra output dir
    fig.savefig(Path(hydra_cfg['runtime']['output_dir']) / 'test_fig.png')

    known_metrics = (
        metric_1,
        metric_2,
    )

    variables_map = locals().copy()

    metrics_maps = {func.__name__: func for func in known_metrics}

    for met in cfg.get('metrics', []):
        log.info(metrics_maps[met['name']](**{k: variables_map[v] for k, v in met['kwargs'].items()}))


if __name__ == "__main__":
    main()
