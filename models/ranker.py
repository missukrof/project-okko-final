import os
import logging
from typing import Dict, List

import catboost as cb
from sklearn.utils import class_weight
import numpy as np
import pandas as pd

from configs.config import settings
from utils.utils import (
    add_model_to_zip, 
    load_model_from_zip
)

class Ranker:
    def __init__(self, is_infer=True):
        if is_infer:
            logging.info("Loading ranker model...")
            load_model_from_zip(path_zip=settings.CBM_TRAIN_PARAMS.MODEL_PATH_ZIP)
            self.ranker = cb.CatBoostClassifier().load_model(
                fname=settings.CBM_TRAIN_PARAMS.MODEL_PATH
            )
            os.remove(settings.CBM_TRAIN_PARAMS.MODEL_PATH)
        else:
            pass

    @staticmethod
    def fit(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
    ) -> None:
        """
        trains catboost clf model
        :X_train:
        :y_train:
        :X_test:
        :y_test:
        :ranker_params
        """
        if X_test is None and y_test is None:
            all_y = y_train
        else:
            all_y = pd.concat([y_train, y_test], ignore_index=True)
        
        classes = np.unique(all_y)
        class_weights = dict(zip(classes, list(
            class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=all_y)
            )))
        logging.info(f"Class weights: {class_weights}")


        logging.info(f"Init ranker model...")
        cbm_classifier = cb.CatBoostClassifier(
            loss_function=settings.CBM_TRAIN_PARAMS.LOSS_FUNCTION,
            class_weights=class_weights,
            iterations=settings.CBM_TRAIN_PARAMS.ITERATIONS,
            learning_rate=settings.CBM_TRAIN_PARAMS.LEARNING_RATE,
            depth=settings.CBM_TRAIN_PARAMS.DEPTH,
            random_state=settings.CBM_TRAIN_PARAMS.RANDOM_STATE,
            verbose=settings.CBM_TRAIN_PARAMS.VERBOSE,
            early_stopping_rounds=settings.CBM_TRAIN_PARAMS.EARLY_STOPPING_ROUNDS,
            cat_features=settings.RANKER_PREPROCESS_FEATURES.CATEGORICAL_COLS,
            allow_writing_files=False
        )

        logging.info("Started fitting the model and choosing features...")
        summary = cbm_classifier.select_features(
            X_train, y_train,
            eval_set=(X_test, y_test),
            features_for_select=list(range(X_train.shape[1])),
            num_features_to_select=25,
            steps=1,
            algorithm=cb.EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=cb.EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Verbose',
            plot=False
        )

        print(f"Selected features: {summary.get('selected_features_names')}")

        cbm_classifier.save_model(settings.CBM_TRAIN_PARAMS.MODEL_PATH)
        add_model_to_zip(path_model=settings.CBM_TRAIN_PARAMS.MODEL_PATH,
                         path_zip=settings.CBM_TRAIN_PARAMS.MODEL_PATH_ZIP)

    def infer(self, ranker_input: List) -> Dict[str, int]:
        """
        inference for the output from lfm model
        :user_id:
        :candidates: dict with ranks {"item_id": 1, ...}
        """

        logging.info("Making predictions...")
        preds = self.ranker.predict_proba(ranker_input)[:, 1]

        return preds
