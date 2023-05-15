import pandas as pd

import sys
sys.path.append('C:/Users/a.kuznetsova/Documents/Python Scripts/okko/project-okko-team-work-final/project-okko-final')

from configs.config import settings
from utils.utils import (
    create_dataframes,
    dump_parquet_to_local_path
    )

from fire import Fire


def generate_fav_user_person(df, user_col, feature_col):
    count = df.copy()
    count[feature_col] = count[feature_col]\
        .fillna({i: ['Unknown'] for i in count.index})
    count = count[[user_col, feature_col]]\
        .explode(feature_col)\
            .groupby([user_col, feature_col])\
                .size()\
                    .reset_index()
    count = count.rename(columns={0: 'count'})\
        .sort_values(by=[user_col, 'count'], ascending=False)
    count = count[count[feature_col] != 'Unknown']
    count = count.groupby(by=user_col).head(1)

    return count[[user_col, feature_col]].rename(columns={feature_col: f'fav_{feature_col}'})


def generate_fav_user_feature(df, user_col, feature_col):
    temp_df = df.copy()
    temp_df[feature_col] = temp_df[feature_col]\
        .fillna({i: ['Unknown'] for i in temp_df.index})
    temp_df['count'] = temp_df[feature_col].apply(len)
    temp_df['watch_duration_minutes'] /= temp_df['count']

    temp_df = temp_df[[user_col, feature_col, 'watch_duration_minutes']]\
        .explode(feature_col)\
            .groupby([user_col, feature_col])\
                .agg('mean')\
                    .reset_index()
    temp_df = temp_df.sort_values(by=[user_col, 'watch_duration_minutes'], ascending=False)
    temp_df = temp_df[temp_df[feature_col] != 'Unknown']
    temp_df = temp_df.groupby(by=user_col).head(1)

    return temp_df[[user_col, feature_col]].rename(columns={feature_col: f'fav_{feature_col}'})


def generate_users_kids(df, user_col, feature_col):

    user_watched_sum = df[[user_col, 'watch_duration_minutes']]\
        .groupby([user_col])\
            .agg('sum')\
                .reset_index()
    user_watched_sum_by_age = df[[user_col, feature_col, 'watch_duration_minutes']]\
        .groupby([user_col, feature_col])\
            .agg('sum')\
                .reset_index()
    merged_user_watched = user_watched_sum_by_age.merge(
        user_watched_sum, 
        on=[user_col], 
        how='left', 
        suffixes=('_sum_by_age', '_sum')
        )
    merged_user_watched['perc'] = (merged_user_watched['watch_duration_minutes_sum_by_age'] / merged_user_watched['watch_duration_minutes_sum']) * 100
    merged_user_watched = merged_user_watched[merged_user_watched[feature_col] <= 6.0]
    merged_user_watched = merged_user_watched[[user_col, 'perc']]\
        .groupby([user_col])\
            .agg('sum')\
                .reset_index()
    merged_user_watched = merged_user_watched[merged_user_watched['perc'] >= 50.0]
    merged_user_watched['kids_flg'] = [1] * len(merged_user_watched[user_col].unique())
    
    return merged_user_watched[[user_col, 'kids_flg']]


def generation_users_metadata():

    interactions, _, interactions_merged = create_dataframes()

    users_metadata = pd.DataFrame({
        settings.USERS_FEATURES.USER_IDS: interactions[settings.USERS_FEATURES.USER_IDS].unique()
        })

    users_metadata = users_metadata.merge(
        generate_fav_user_person(
        interactions_merged, 
        settings.USERS_FEATURES.USER_IDS, 
        'actors'
        ),
          on=[settings.USERS_FEATURES.USER_IDS], how='left')
    users_metadata = users_metadata.merge(
        generate_fav_user_person(
        interactions_merged, 
        settings.USERS_FEATURES.USER_IDS, 
        'director'
        ), on=[settings.USERS_FEATURES.USER_IDS], how='left')

    users_metadata = users_metadata.merge(
        generate_fav_user_feature(
        interactions_merged, 
        settings.USERS_FEATURES.USER_IDS, 
        'genres'
        ), on=[settings.USERS_FEATURES.USER_IDS], how='left')
    users_metadata = users_metadata.merge(
        generate_fav_user_feature(
        interactions_merged, 
        settings.USERS_FEATURES.USER_IDS, 
        'country'
        ), on=[settings.USERS_FEATURES.USER_IDS], how='left')

    users_metadata = users_metadata.merge(
        generate_users_kids(
        interactions_merged, 
        settings.USERS_FEATURES.USER_IDS, 
        'age_rating'
        ), on=[settings.USERS_FEATURES.USER_IDS], how='left')
    users_metadata['kids_flg'] = users_metadata['kids_flg'].fillna(0)

    dump_parquet_to_local_path(
        data=users_metadata,
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER,
        file_name=settings.DATA_FILES_G.USERS_METADATA_FILE
        )


if __name__ == '__main__':
    Fire(
    {
        'generation_users_metadata': generation_users_metadata
        }
    )