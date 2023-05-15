from data_prep.initial_data_preprocessing import (
    first_data_copy,
    generate_datetime_feature,
    fill_unknown_movie_ids,
    fill_series_duration_nans,
    generalization_of_entity_types,
    convert_to_list_type
)

from data_prep.feature_engineering_movies import (
    convert_to_datetime_type,
    generate_dummy_cols,
    generate_main_features,
    fill_nans_by_column,
    generate_movie_popularity,
    generate_year_features,
    generate_watch_duration_minutes_mean
)

from data_prep.embeddings_movies import (
    add_embeddings_to_dataframe
)

from data_prep.feature_engineering_interactions import (
    generate_watch_features
)

from data_prep.feature_engineering_users import (
    generation_users_metadata
)

import sys
sys.path.append('C:/Users/a.kuznetsova/Documents/Python Scripts/okko/project-okko-team-work-final/project-okko-final')

from configs.config import settings
from utils.utils import (
    copy_files_to_path
)

from fire import Fire


def final_data_copy() -> None:
    for file in settings.DATA_FILES_P.values() + settings.DATA_FILES_G.values():
        copy_files_to_path(
            src_file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
            dst_file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER,
            file_name=file
            )


def perform_preprocessing():
    first_data_copy()
    generate_datetime_feature()
    fill_unknown_movie_ids()
    fill_series_duration_nans()
    generalization_of_entity_types()
    convert_to_list_type()
    convert_to_datetime_type()
    generate_dummy_cols()
    generate_main_features()
    fill_nans_by_column()
    generate_movie_popularity()
    generate_year_features()
    generate_watch_duration_minutes_mean()
    generate_watch_features()
    generation_users_metadata()
    add_embeddings_to_dataframe()
    final_data_copy()


if __name__ == '__main__':
    Fire(
    {
        'perform_preprocessing': perform_preprocessing
        }
    )