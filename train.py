import pandas as pd

from models.lfm import LFMModel
from models.ranker import Ranker
from utils.utils import read_parquet
from data_prep.prepare_ranker_data import prepare_data_for_train

from configs.config import settings

from fire import Fire

import logging

def train_lfm(data_path: str = None) -> None:
    """
    trains model for a given data with interactions
    :data_path: str, path to parquet with interactions
    """

    data = read_parquet(
        file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
        )

    logging.info('Started training LightFM model...')
    lfm = LFMModel(is_infer = False) # train mode
    lfm.fit(
        data,
        user_col=settings.USERS_FEATURES.USER_IDS,
        item_col=settings.MOVIES_FEATURES.MOVIE_IDS
    )
    logging.info('Finished training LightFM model!')

def train_ranker():
    """
    executes training pipeline for 2nd level model
    all params are stored in configs
    """

    X_train, X_test, y_train, y_test = prepare_data_for_train()
    ranker = Ranker(is_infer = False) # train mode
    ranker.fit(X_train, y_train, X_test, y_test)
    logging.info('Finished training Ranker model!')

if __name__ == '__main__':
    Fire(
    {
        'train_lfm': train_lfm,
        'train_cbm': train_ranker
        }
    )