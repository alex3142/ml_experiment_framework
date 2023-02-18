from typing import (
    Dict,
    Any,
    Optional,
)
from pathlib import Path
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

import pandas as pd
import yaml


def parse_config(experiment_dir: Optional[str], config_file: str) -> Dict[str, Any]:
    """
    Parse the configuration file for the experiment
    :param experiment_dir:  Optional[str] - directory of experiment configuration files
    :param config_file: str - configuration filename
    :return: Dict[str, Any] loaded configuration file for experiment
    """

    if experiment_dir is None:
        experiment_dir = Path(__file__).parent.parent.joinpath('experiment_configs')
    else:
        experiment_dir = Path(experiment_dir)

    config_file_full = experiment_dir.joinpath(config_file)
    config = yaml.load(open(config_file_full).read(), Loader=yaml.FullLoader)
    config['config_filename'] = str(config_file_full)
    logger.info('config loaded.')

    return config


def load_data(
        filename: str,
        load_data_kwargs: Dict[str, Any],
        file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data from filename
    :param filename: str - filename to load data from
    :return: pd.DataFrame - loaded data
    """

    if file_path is None:
        file_path = Path(__file__).parent.parent.parent.joinpath('data').joinpath('processed')

    file_path = file_path.joinpath(filename)

    data = pd.read_csv(file_path, **load_data_kwargs)
    logger.info('data loaded.')
    return data
