import pandas as pd

import logging
from configs.config import settings
from utils.utils import (
    create_dataframes,
    dump_parquet_to_local_path
    )

from fire import Fire


def generate_watch_features():

    logging.info("Cenerating interactions watching features...")
    interactions, _, interactions_merged = create_dataframes()

    interactions_merged['watched_pct'] = (interactions_merged['watch_duration_minutes'] / interactions_merged['duration']) * 100
    interactions['watched_pct'] = interactions_merged['watched_pct']

    interactions['watch_duration_minutes_cumsum'] = interactions_merged[(interactions_merged['entity_type'] == 'Сериал')]\
        .groupby([settings.USERS_FEATURES.USER_IDS, settings.MOVIES_FEATURES.MOVIE_IDS])\
            .cumsum()['watch_duration_minutes']
    interactions['watched_pct_cumsum'] = interactions_merged[(interactions_merged['entity_type'] == 'Сериал')]\
        .groupby([settings.USERS_FEATURES.USER_IDS, settings.MOVIES_FEATURES.MOVIE_IDS])\
            .cumsum()['watched_pct']
    
    interactions['watch_duration_minutes_cumsum'] = interactions['watch_duration_minutes_cumsum']\
        .fillna(interactions['watch_duration_minutes'])
    interactions['watched_pct_cumsum'] = interactions['watched_pct_cumsum']\
        .fillna(interactions['watched_pct'])
    
    dump_parquet_to_local_path(
        data=interactions,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
        )


if __name__ == '__main__':
    Fire(
    {
        'generate_watch_features': generate_watch_features
        }
    )