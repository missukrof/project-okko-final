import os
import ast
import pandas as pd
from typing import Dict, List

from configs.config import settings
from utils.utils import (
    save_pickle,
    copy_files_to_path,
    create_dataframes,
    read_parquet_from_local_path,
    dump_parquet_to_local_path
    )
from data_prep.get_missing_data_from_web import (
    find_missing_titles
    )

from fire import Fire


def first_data_copy() -> None:
    for file in settings.DATA_FILES_P.values():
        copy_files_to_path(
            src_file_folder=settings.DATA_FOLDERS.INITIAL_DATA_FOLDER,
            dst_file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
            file_name=file
            )


def generate_datetime_feature() -> None:
    
    interactions, _, _ = create_dataframes()
    
    date_columns = ['year', 'month', 'day']

    interactions = interactions.astype(
        dict(zip(date_columns, ['str'] * len(date_columns)))
        )

    interactions['datetime'] = pd.to_datetime(
        interactions['year'] + '-' + interactions['month'] + '-' + interactions['day'],
        format='%Y-%m-%d'
        )

    interactions = interactions.astype(
        dict(zip(date_columns, ['int64'] * len(date_columns)))
        )
    
    dump_parquet_to_local_path(
        data=interactions,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
        )


def fill_unknown_movie_ids() -> None:
    
    _, movies_metadata, interactions_merged = create_dataframes()
    interactions_merged_all_nans = interactions_merged[interactions_merged[interactions_merged.columns[7:]].isna().all(axis=1)]
    missing_ids_list = interactions_merged_all_nans[settings.MOVIES_FEATURES.MOVIE_IDS].unique()

    missing_ids_names_mapping = find_missing_titles(missing_ids_list)
    error_movies = {}

    for k, v in missing_ids_names_mapping.items():
        temp_movie_features_df = movies_metadata[movies_metadata['title'] == v]
        if temp_movie_features_df.shape[0] > 0:
            movie_features = temp_movie_features_df.iloc[0].to_list()
            movie_features[0] = k
            movie_features[-1] = temp_movie_features_df['duration'].sum()
            movies_metadata = movies_metadata.append(pd.Series(movie_features, index=movies_metadata.columns), ignore_index=True)
        else:
            error_movies[k] = v

        del temp_movie_features_df
    
    if len(error_movies) > 0:

        parent_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        save_pickle(
            data=error_movies, 
            path=f"{parent_directory}{settings.NAN_MAPPING_PATH.FOLDER}{settings.NAN_MAPPING_PATH.ERRORS_FILE_NAME}"
        )

    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )
    

def fill_series_duration_nans() -> None:
    
    _, movies_metadata, _ = create_dataframes()

    total_series_time = movies_metadata[movies_metadata['entity_type'] == 'Серия']\
        .groupby('title')['duration']\
            .sum()\
                .to_dict()
    total_series_time_indexes = movies_metadata[(movies_metadata['entity_type'] == 'Сериал') & \
                                                (movies_metadata['duration'].isnull())]['duration']\
                                                    .fillna(movies_metadata['title']\
                                                            .map(total_series_time))\
                                                                .to_dict()

    movies_metadata = movies_metadata.reset_index()
    movies_metadata['duration'] = movies_metadata['duration'].fillna(movies_metadata['index'].map(total_series_time_indexes))
    movies_metadata = movies_metadata.drop(['index'], axis=1)

    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


def generalization_of_entity_types() -> None:
    
    _, movies_metadata, interactions_merged = create_dataframes()

    for ind, row in interactions_merged[interactions_merged['entity_type'].isin(['Серия', 'Сезон'])].iterrows():

        temp_serial_data = movies_metadata[
            (movies_metadata['title'] == row['title']) & 
            (movies_metadata['entity_type'] == 'Сериал') & 
            (movies_metadata['actors'] == row['actors']) & 
            (movies_metadata['director'] == row['director']) & 
            (movies_metadata['country'] == row['country']) & 
            (movies_metadata['age_rating'] == row['age_rating'])
            ]
        try:
            interactions_merged.loc[ind, 'movie_id'] = temp_serial_data['movie_id'].values[0]
        except IndexError:
            continue

    interactions = interactions_merged[settings.INITIAL_FEATURES.INTERACTIONS + settings.PREPROCESSED_FEATURES.INTERACTIONS]
    
    dump_parquet_to_local_path(
        data=interactions,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
        )


def convert_to_list_type(
    columns: List[str] = settings.MOVIES_FEATURES.LIST_MOVIE_FEATURES
    ) -> None:

    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    for col in columns:
        movies_metadata[col] = movies_metadata[col].apply(lambda x: None if x == None else ast.literal_eval(x))

    dump_parquet_to_local_path(
        data=movies_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )


if __name__ == '__main__':
    Fire(
    {
        'first_data_copy': first_data_copy,
        'generate_datetime_feature': generate_datetime_feature,
        'fill_unknown_movie_ids': fill_unknown_movie_ids,
        'fill_series_duration_nans': fill_series_duration_nans,
        'generalization_of_entity_types': generalization_of_entity_types,
        'convert_to_list_type': convert_to_list_type,
        }
    )