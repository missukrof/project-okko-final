import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from configs.config import settings
from utils.utils import (
    generate_lightfm_recs_mapper,
    load_model,
    read_parquet
)


def get_ranker_sample(preds: pd.DataFrame, users_sample: pd.DataFrame):
    """
    final step to use candidates generation and users interaction to define
    train data - join features, define target, split into train & test samples
    """
    local_test = users_sample.copy(deep=True)

    # prepare train & test
    positive_preds = pd.merge(preds, local_test, how="inner", 
                              on=[settings.USERS_FEATURES.USER_IDS, 
                                  settings.MOVIES_FEATURES.MOVIE_IDS])
    positive_preds[settings.RANKER_PREPROCESS_FEATURES.TARGET] = 1
    logging.info(f"Shape of the positive target preds: {positive_preds.shape}")

    negative_preds = pd.merge(preds, local_test, how="left", 
                              on=[settings.USERS_FEATURES.USER_IDS, 
                                  settings.MOVIES_FEATURES.MOVIE_IDS])
    negative_preds = negative_preds.loc[negative_preds["watched_pct"].isnull()].sample(
        frac=settings.RANKER_DATA.NEG_FRAC
    )
    negative_preds[settings.RANKER_PREPROCESS_FEATURES.TARGET] = 0
    logging.info(f"Shape of the negative target preds: {negative_preds.shape}")

    # random split to train ranker
    train_users, test_users = train_test_split(
        local_test[settings.USERS_FEATURES.USER_IDS].unique(),
        test_size=0.2,
        random_state=settings.RANKER_DATA.RANDOM_STATE,
    )

    cbm_train_set = shuffle(
        pd.concat(
            [
                positive_preds.loc[positive_preds[settings.USERS_FEATURES.USER_IDS].isin(train_users)],
                negative_preds.loc[negative_preds[settings.USERS_FEATURES.USER_IDS].isin(train_users)],
            ]
        )
    )

    cbm_test_set = shuffle(
        pd.concat(
            [
                positive_preds.loc[positive_preds[settings.USERS_FEATURES.USER_IDS].isin(test_users)],
                negative_preds.loc[negative_preds[settings.USERS_FEATURES.USER_IDS].isin(test_users)],
            ]
        )
    )

    users_data = read_parquet(
        file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
        file_name=settings.DATA_FILES_G.USERS_METADATA_FILE
        )
    items_data = read_parquet(
        file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    # join user features
    cbm_train_set = pd.merge(
        cbm_train_set,
        users_data[[settings.USERS_FEATURES.USER_IDS] + settings.PREPROCESSED_FEATURES.USER_METADATA],
        how="left",
        on=[settings.USERS_FEATURES.USER_IDS],
    )
    cbm_test_set = pd.merge(
        cbm_test_set,
        users_data[[settings.USERS_FEATURES.USER_IDS] + settings.PREPROCESSED_FEATURES.USER_METADATA],
        how="left",
        on=[settings.USERS_FEATURES.USER_IDS],
    )
    # join item features
    cbm_train_set = pd.merge(
        cbm_train_set,
        items_data[[settings.MOVIES_FEATURES.MOVIE_IDS] + settings.PREPROCESSED_FEATURES.MOVIES_METADATA],
        how="left",
        on=[settings.MOVIES_FEATURES.MOVIE_IDS],
    )
    cbm_test_set = pd.merge(
        cbm_test_set,
        items_data[[settings.MOVIES_FEATURES.MOVIE_IDS] + settings.PREPROCESSED_FEATURES.MOVIES_METADATA],
        how="left",
        on=[settings.MOVIES_FEATURES.MOVIE_IDS],
    )

    # final steps
    drop_cols = (
        [settings.USERS_FEATURES.USER_IDS] 
        + [settings.MOVIES_FEATURES.MOVIE_IDS]
        + [settings.RANKER_PREPROCESS_FEATURES.TARGET]
        + settings.RANKER_PREPROCESS_FEATURES.DROP_FEATURES
    )

    X_train, y_train = (
        cbm_train_set.drop(
            drop_cols,
            axis=1,
            errors='ignore'
        ),
        cbm_train_set[settings.RANKER_PREPROCESS_FEATURES.TARGET],
    )
    X_test, y_test = (
        cbm_test_set.drop(
            drop_cols,
            axis=1,
            errors='ignore'
        ),
        cbm_test_set[settings.RANKER_PREPROCESS_FEATURES.TARGET],
    )
    logging.info(f"X_train set shape: {X_train.shape}, X_test validate set shape: {X_test.shape}")

    # no time dependent feature -- we can leave it with mode
    X_train = X_train.fillna(X_train.mode().iloc[0])
    X_test = X_test.fillna(X_test.mode().iloc[0])

    return X_train, X_test, y_train, y_test


def prepare_data_for_train() -> Tuple[pd.DataFrame]:
    """
    function to prepare data to train catboost classifier.
    Basically, you have to wrap up code from full_recsys_pipeline.ipynb
    where we prepare data for classifier. In the end, it should work such
    that we trigger and use fit() method from ranker.py

        paths_config: dict, wher key is path name and value is the path to data
    """
    # load model artefacts
    model = load_model(settings.LFM_TRAIN_PARAMS.MODEL_PATH)
    dataset = load_model(settings.LFM_TRAIN_PARAMS.MAPPER_PATH)

    users_sample = read_parquet(
        file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
        )
    
    # pred candidates
    preds = users_sample[[settings.USERS_FEATURES.USER_IDS]].drop_duplicates().reset_index(drop=True)

    # init mapper with model
    item_ids = list(dataset.mapping()[2].values())
    item_inv_mapper = {v: k for k, v in dataset.mapping()[2].items()}
    mapper = generate_lightfm_recs_mapper(
        model,
        item_ids=item_ids,
        known_items=dict(),
        N=settings.LFM_PREDS_PARAMS.TOP_K,
        user_features=None,
        item_features=None,
        user_mapping=dataset.mapping()[0],
        item_inv_mapping=item_inv_mapper,
        num_threads=20,
    )
    preds[settings.MOVIES_FEATURES.MOVIE_IDS] = preds[settings.USERS_FEATURES.USER_IDS].map(mapper)

    # define target & prepare ranker sample
    preds = preds.explode(settings.MOVIES_FEATURES.MOVIE_IDS)
    logging.info(f"Shape of the preds: {preds.shape}")

    preds["rank"] = preds.groupby(settings.USERS_FEATURES.USER_IDS).cumcount() + 1
    logging.info(
        f"Number of unique candidates generated by the LFM: {preds[settings.MOVIES_FEATURES.MOVIE_IDS].nunique()}"
    )

    train_data = get_ranker_sample(preds=preds, users_sample=users_sample)

    return train_data


def get_items_features(item_ids: List[int], item_cols: List[str]) -> Dict[int, Any]:
    """
    function to get items features from our available data
    that we used in training (for all candidates)
        :item_ids:  item ids to filter by
        :item_cols: feature cols we need for inference

    EXAMPLE OUTPUT
    {
    9169: {
    'content_type': 'film',
    'release_year': 2020,
    'for_kids': None,
    'age_rating': 16
        },

    10440: {
    'content_type': 'series',
    'release_year': 2021,
    'for_kids': None,
    'age_rating': 18
        }
    }

    """
    item_features = read_parquet(
        file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )

    item_features = item_features.set_index(settings.MOVIES_FEATURES.MOVIE_IDS)
    item_features = item_features.to_dict("index")

    # collect all items
    output = {}
    for id in item_ids:
        output[id] = {k: v for k, v in item_features.get(id).items() if k in item_cols}

    return output


def get_user_features(user_id: int, user_cols: List[str]) -> Dict[str, Any]:
    """
    function to get user features from our available data
    that we used in training
        :user_id: user id to filter by
        :user_cols: feature cols we need for inference

    EXAMPLE OUTPUT
    {
        'age': None,
        'income': None,
        'sex': None,
        'kids_flg': None
    }
    """
    users = read_parquet(
        file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
        file_name=settings.DATA_FILES_G.USERS_METADATA_FILE
        )
    users = users.set_index(settings.USERS_FEATURES.USER_IDS)
    users_dict = users.to_dict("index")
    return {k: v for k, v in users_dict.get(user_id).items() if k in user_cols}


def prepare_ranker_input(
    candidates: Dict[int, int],
    item_features: Dict[int, Any],
    user_features: Dict[int, Any],
    ranker_features_order,
):
    ranker_input = []
    for k in item_features.keys():
        item_features[k].update(user_features)
        item_features[k]["rank"] = candidates[k]
        item_features[k] = {
            feature: item_features[k][feature] for feature in ranker_features_order
        }
        ranker_input.append(list(item_features[k].values()))

    return ranker_input
