import pandas as pd
import numpy as np

from models.lfm import LFMModel
from models.ranker import Ranker

from configs.config import settings
from utils.utils import (
    read_parquet,
    compute_metrics
)
from data_prep.prepare_ranker_data import (
    get_user_features,
    get_items_features,
    prepare_ranker_input
)

import logging


def get_recommendations(user_id: int, top_k: int = 20):
    """
    function to get recommendation for a given user id
    """

    lfm_model = LFMModel()
    ranker = Ranker()

    try:
        logging.info('Getting 1st level candidates...')
        candidates = lfm_model.infer(user_id = user_id, top_k = top_k)

        logging.info('Getting features...')
        user_features = get_user_features(user_id, user_cols=settings.PREPROCESSED_FEATURES.USER_METADATA)
        item_features = get_items_features(item_ids = list(candidates.keys()), item_cols = settings.PREPROCESSED_FEATURES.MOVIES_METADATA)

        ranker_input = prepare_ranker_input(
            candidates = candidates,
            item_features = item_features,
            user_features=user_features,
            ranker_features_order=ranker.ranker.feature_names_
            )
        preds = ranker.infer(ranker_input = ranker_input)

        item_features = read_parquet(
            file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
            file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
            )[[settings.MOVIES_FEATURES.MOVIE_IDS, "title"]]
        
        # output = dict(zip(candidates.keys(), preds))
        output = pd.DataFrame({settings.MOVIES_FEATURES.MOVIE_IDS: candidates.keys(), "cbm_preds": preds})
        output = output.merge(item_features, on=[settings.MOVIES_FEATURES.MOVIE_IDS], how="left")
        output = output.sort_values(by=["cbm_preds"], ascending=[False])
        output["temp"] = output.shape[0] * [1]
        output["cbm_rank"] = output.groupby("temp").cumcount() + 1
        output.drop(["temp"], axis=1, inplace=True)

        cols = output.columns.tolist()
        cols = [cols[ind] for ind in [0, 2, 1, 3]]
        output = output[cols]

        interactions = read_parquet(
            file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
            file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
            )
        
        output[settings.USERS_FEATURES.USER_IDS] = [user_id] * output.shape[0]

        metrics = compute_metrics(
            df_true=interactions[
                interactions[settings.USERS_FEATURES.USER_IDS] == user_id][[settings.USERS_FEATURES.USER_IDS, 
                                                                            settings.MOVIES_FEATURES.MOVIE_IDS]],
            df_pred=output,
            K=10,
            rank_col='cbm_rank'
            )

        output.drop([settings.USERS_FEATURES.USER_IDS], axis=1, inplace=True)

        return output.reset_index(drop=True), metrics
    
    except KeyError:
        output = "<p>Please enter the User ID that is in OKKO database, {} is irrelevant.</p>".format(user_id)

        return output

# if __name__ == '__main__':

#     users_data = read_parquet(
#         file_folder=settings.DATA_FOLDERS.PREPROCESSED_DATA_FOLDER, 
#         file_name=settings.DATA_FILES_G.USERS_METADATA_FILE
#         )
    
#     random_user_id = np.random.choice(users_data[settings.USERS_FEATURES.USER_IDS].unique(), 1)[0]
#     print(random_user_id)
#     print(get_recommendations(random_user_id))