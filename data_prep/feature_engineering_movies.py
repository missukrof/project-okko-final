import math
import numpy as np
import pandas as pd
from typing import List
from transliterate import translit

import logging
from configs.config import settings
from utils.utils import (
    create_dataframes,
    read_parquet_from_local_path,
    dump_parquet_to_local_path
    )

from fire import Fire


def convert_to_datetime_type(
        feature_cols: List[str] = ['release_world']
        ) -> None:

    logging.info("Converting movies metadata columns to datetime type...")
    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    for column in feature_cols:
        movies_metadata[column] = pd.to_datetime(movies_metadata[column])

    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


def generate_dummy_cols(
        feature_col: List[str] = ['age_rating', 'genres', 'country', 'entity_type'],
        list_cols: List[str] = ['genres', 'country']
        ) -> None:

    logging.info("Cenerating movies dummy cols...")
    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )
    
    for column in feature_col:
        
        if column not in list_cols:
            dummy_metadata = pd.get_dummies(movies_metadata[column])
            
            if column == 'age_rating':
                dummy_metadata = pd.get_dummies(movies_metadata[column])
                dummy_metadata['for_kids'] = dummy_metadata.loc[:, [0.0, 6.0]].sum(axis=1)
                dummy_metadata = dummy_metadata.rename({18.0: 'for_adults'}, axis=1)
        
        elif column in list_cols:
            dummy_metadata = pd.get_dummies(movies_metadata[column].explode()).sum(level=0)
            cleaned_columns = [col for col in dummy_metadata.columns if col not in ['\t', 'test']]
            dummy_metadata = dummy_metadata[cleaned_columns]
        
        movies_metadata = pd.concat([movies_metadata, dummy_metadata], axis=1)
        del dummy_metadata
    
    movies_metadata.rename(columns=dict(zip(
        movies_metadata.columns.to_list(), 
        [translit(str(x), 'ru', reversed=True) for x in movies_metadata.columns.to_list()]
        )),
        inplace=True
        )

    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


def generate_main_features(
        feature_cols: List[str] = ['actors', 'director', 'genres', 'country']
        ) -> None:

    logging.info("Cenerating main features from movies metadata...")
    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    for column in feature_cols:
        movies_metadata[f'main_{column}'] = movies_metadata[column].apply(
            lambda x: None if type(x) != np.ndarray else x[0]
            )
    
    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


def fill_nans_by_column(
        feature_cols: List[str] = ['actors', 'director', 'genres', 'country']
        ) -> None:

    logging.info("Filling categorical missing values...")
    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    for column in feature_cols:

        movies_metadata[column] = movies_metadata[column].fillna(
            {
                i: [f'Unknown_{column}'] for i in movies_metadata.index
                }
            )
    
    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


def generate_movie_popularity(
        grouby_cols: List[str] = ['movie_id', 'genres', 'country']
        ) -> None:

    logging.info("Cenerating movie popularity...")
    interactions, movies_metadata, interactions_merged = create_dataframes()

    for column in grouby_cols:

        if type(interactions_merged[column][0]) is not str:

            interactions_merged['ordered'] = interactions_merged[column].apply(
                lambda x: ', '.join(sorted(x))
                )
            popularity = interactions_merged.groupby('ordered')\
                .size().reset_index(name=f'popularity_{column}')
            popularity.loc[
                popularity[popularity['ordered'].str.match('^Unknown_*') == True]\
                    .index.values[0], 
                f'popularity_{column}'
                ] = np.NaN
            movies_metadata['ordered'] = movies_metadata[column].apply(lambda x: ', '.join(sorted(x)))
            interactions_merged.drop(['ordered'], axis=1, inplace=True)
            output = movies_metadata.merge(popularity, on='ordered', how='left')
            output.drop(['ordered'], axis=1, inplace=True)
        
        else:
            
            popularity = interactions_merged.groupby(column).size().reset_index(name=f'popularity_{column}')
            
            try:
                popularity.loc[
                    popularity[popularity[column].str.match('^Unknown_*') == True].index.values[0], 
                    f'popularity_{column}'
                    ] = np.NaN
            except IndexError:
                pass

            output = movies_metadata.merge(popularity, on=column, how='left')
    
    dump_parquet_to_local_path(
        data=output,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


def generate_year_features(
        datetime_cols: List[str] = ['release_world']
        ) -> None:

    logging.info("Cenerating movie categorical year features...")
    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    for column in datetime_cols:
        
        movies_metadata[f'year_{column}'] = movies_metadata[column].dt.year
        movies_metadata[f'year_cat_{column}'] = movies_metadata[f'year_{column}'].apply(
            lambda x: str(math.floor(x / 10) * 10) + 's' if not math.isnan(x) else x
            )
        movies_metadata[f'year_cat_{column}'] = movies_metadata[f'year_cat_{column}'].fillna(f'Unknown_{column}')
        year_cat = movies_metadata[f'year_cat_{column}']
        movies_metadata = pd.get_dummies(movies_metadata, columns=[f'year_cat_{column}'])
        movies_metadata[f'year_cat_{column}'] = year_cat
    
    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )



def generate_watch_duration_minutes_mean(
        feature_cols: List[str] = ['movie_id']
        ) -> None:

    logging.info("Cenerating mean watch duration minutes by movie...")
    _, movies_metadata, interactions_merged = create_dataframes()

    for column in feature_cols:
        movies_metadata = movies_metadata.merge(interactions_merged[[column, 'watch_duration_minutes']]\
                                                .groupby(column).agg('mean')\
                                                .reset_index()\
                                                .rename(columns={
                                                    'watch_duration_minutes': f'watch_duration_minutes_mean_by_{column}'
                                                    }), 
                                                on=[column], how='left')
    
    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


if __name__ == '__main__':
    Fire(
    {
        'convert_to_datetime_type': convert_to_datetime_type,
        'generate_dummy_cols': generate_dummy_cols,
        'generate_main_features': generate_main_features,
        'fill_nans_by_column': fill_nans_by_column,
        'generate_movie_popularity': generate_movie_popularity,
        'generate_year_features': generate_year_features,
        'generate_watch_duration_minutes_mean': generate_watch_duration_minutes_mean
        }
    )