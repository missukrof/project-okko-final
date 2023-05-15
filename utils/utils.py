import os
import shutil
import dill
import pickle
import toml
import numpy as np
import pandas as pd

import sys
sys.path.append('C:/Users/a.kuznetsova/Documents/Python Scripts/okko/project-okko-team-work-final/project-okko-final')

from configs.config import settings


def copy_files_to_path(src_file_folder, dst_file_folder, file_name):
    """
    Написать документацию!!!
    """
    parent_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

    source_path = f"{parent_directory}{src_file_folder}{file_name}"
    destination_path = f"{parent_directory}{dst_file_folder}{file_name}"

    shutil.copy(source_path, destination_path)


def read_parquet(file_folder, file_name):
    parent_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    data = pd.read_parquet(f"{parent_directory}{file_folder}{file_name}", engine='pyarrow')
    return data

def read_parquet_from_local_path(file_folder, file_name):
    """
    Написать документацию!!!
    """
    parent_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    data = read_parquet(file_folder=file_folder, file_name=file_name)
    toml_file_name = "features_1.toml"

    with open(f"{parent_directory}\configs\{toml_file_name}", "r") as toml_file:
        features = toml.load(toml_file)
    
    if file_name.split(".")[0].upper() == 'USERS_METADATA':
        features['PREPROCESSED_FEATURES'][f'{file_name.split(".")[0]}'.upper()] = data.columns.to_list()
    else:
        try:
            features['PREPROCESSED_FEATURES'][f'{file_name.split(".")[0]}'.upper()] = list(set(data.columns.to_list()) \
                - set(features['INITIAL_FEATURES'][f'{file_name.split(".")[0]}'.upper()]))
        except KeyError:
            features['INITIAL_FEATURES'][f'{file_name.split(".")[0]}'.upper()] = data.columns.to_list()

    with open(f"{parent_directory}\configs\{toml_file_name}", "w") as toml_file:
        toml.dump(features, toml_file)

    return data


def dump_parquet_to_local_path(data, file_folder, file_name):
    """
    Написать документацию!!!
    """
    parent_directory = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    data.to_parquet(f"{parent_directory}{file_folder}{file_name}", index=False, engine='pyarrow')


def create_dataframes():

    interactions = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.INTERACTIONS_FILE
        )
    
    movies_metadata = read_parquet_from_local_path(
        file_folder=settings.DATA_FOLDERS.ARTEFACTS_DATA_FOLDER, 
        file_name=settings.DATA_FILES_P.MOVIES_METADATA_FILE
        )
    
    interactions_merged = interactions.merge(movies_metadata, on='movie_id', how='left')

    return interactions, movies_metadata, interactions_merged


# def read_parquet_from_gdrive(url):
#     """
#     gets csv data from a given url (from file -> share -> copy link)
#     :url: *****/view?usp=share_link
#     """
#     file_id = url.split("/")[-2]
#     file_path = "https://drive.google.com/uc?export=download&id=" + file_id
#     data = pd.read_parquet(file_path)

#     return data


def generate_lightfm_recs_mapper(
    model: object,
    item_ids: list,
    known_items: dict,
    user_features: list,
    item_features: list,
    N: int,
    user_mapping: dict,
    item_inv_mapping: dict,
    num_threads: int = 4,
):
    def _recs_mapper(user):
        user_id = user_mapping[user]
        recs = model.predict(
            user_id,
            item_ids,
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )

        additional_N = len(known_items[user_id]) if user_id in known_items else 0
        total_N = N + additional_N
        top_cols = np.argpartition(recs, -np.arange(total_N))[-total_N:][::-1]

        final_recs = [item_inv_mapping[item] for item in top_cols]
        if additional_N > 0:
            filter_items = known_items[user_id]
            final_recs = [item for item in final_recs if item not in filter_items]
        return final_recs[:N]

    return _recs_mapper


def save_model(model: object, path: str):
    with open(f"{path}", "wb") as obj_path:
        dill.dump(model, obj_path)


def load_model(path: str):
    with open(path, "rb") as obj_file:
        obj = dill.load(obj_file)
    return obj


def save_pickle(data: object, path: str):
    with open(f"{path}", "wb") as obj_path:
        pickle.dump(data, obj_path)


def load_pickle(path: str):
    with open(path, "rb") as obj_file:
        obj = pickle.load(obj_file)
    return obj