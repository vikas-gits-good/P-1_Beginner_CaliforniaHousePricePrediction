import os
import pandas as pd
from typing import Literal
from datetime import datetime
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, model_scores
from src.pipeline.pipeline_training import (
    PipelineConstructor,
    # Save_DataFrame,
    MultiModelEstimator,
)
from src.constants.models_params import models_dict, params_dict

from sklearn.pipeline import Pipeline


@dataclass
class ModelTrainerConfig:
    ppln_save_path: str = "artifacts/03_Pipeline"
    model_save_path: str = "artifacts/04_Models"
    scores_save_path: str = "artifacts/05_Scores"


class ModelTrainer:
    def __init__(
        self,
        train_path,
        test_path,
        best_model_selection_metric: Literal[
            "r2_score", "f1_score", "accuracy", "recall"
        ] = "r2_score",
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.best_model_metric = best_model_selection_metric

        self.models_dict = models_dict
        self.params_dict = params_dict
        self.md_tr_cfg = ModelTrainerConfig()

    def train_models(self):
        try:
            logging.info("Entered 'train_models' method")
            df_train = pd.read_csv(self.train_path)
            logging.info("Successfully read train dataset")

            df_test = pd.read_csv(self.test_path)
            logging.info("Successfully read test dataset")

            cols_target = "median_house_value"

            x_train = df_train.drop(columns=cols_target)
            y_train = df_train[cols_target]
            x_test = df_test.drop(columns=cols_target)
            y_test = df_test[cols_target]
            logging.info("Successfully created x & y - train & test sets")

            numr_cols = [col for col in x_train.columns if x_train[col].dtypes != "O"]
            catg_cols = [col for col in x_train.columns if x_train[col].dtypes == "O"]
            drop_cols = [
                "longitude",
                "latitude",
                "population",
                "households",
                "total_rooms",
                "total_bedrooms",
            ]
            ordn_cols = [
                [
                    "Less than 1H OCEAN",
                    "NEAR BAY",
                    "NEAR OCEAN",
                    "ISLAND",
                    "INLAND",
                ]
            ]

            pc = PipelineConstructor(
                cols_numr=numr_cols,
                cols_catg=catg_cols,
                cols_drop=drop_cols,
                catg_ordn=ordn_cols,
            )
            ppln_prpc = pc.create_pipeline()
            logging.info("Successfully acquired the training pipeline object")

            ppln_prpc_save = pc.create_pipeline()
            _ = ppln_prpc_save.fit_transform(x_train, y_train)
            ppln_save_path = os.path.join(
                self.md_tr_cfg.ppln_save_path, "ppln_prpc.pkl"
            )
            save_object(file_path=ppln_save_path, obj=ppln_prpc_save)
            logging.info("Pipeline saved to artifacts/03_Pipeline/ppln_prc.pkl")

            ppln_train = Pipeline(
                steps=[
                    ("DataProcessing", ppln_prpc),
                    (
                        "MultiModelEstimator",
                        MultiModelEstimator(
                            models=self.models_dict,
                            param_grids=self.params_dict,
                            cv=3,
                            Method="GridSearchCV",
                        ),
                    ),
                ]
            )
            logging.info("Initiating full pipeline fitting")
            ppln_train.fit(x_train, y_train)
            logging.info(
                f"All {len(self.models_dict.keys())} models successfully fit and ready for testing"
            )

            df_pred, models = ppln_train.predict(x_test)
            df_scores = model_scores(y_true=y_test, y_pred=df_pred)
            logging.info(
                f"All {len(self.models_dict.keys())} models successfully scored on test set"
            )

            best_model_key = (
                df_scores.loc[self.best_model_metric, :]
                .sort_values(ascending=False)
                .index[0]
            )
            # best_model_score = (
            #     df_scores.loc[self.best_model_metric, :]
            #     .sort_values(ascending=False)
            #     .values[0]
            #     * 100
            # )
            # print(best_model_key, best_model_score)
            best_model = models[best_model_key]

            # best_model_name = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_{best_model_key}_{best_model_score:.4f}_%.pkl"
            best_model_name = "best_model.pkl"
            scores_names = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_Scores.pkl"
            best_model_save_path = os.path.join(
                self.md_tr_cfg.model_save_path, best_model_name
            )
            scores_save_path = os.path.join(
                self.md_tr_cfg.scores_save_path, scores_names
            )
            save_object(file_path=best_model_save_path, obj=best_model)
            save_object(file_path=scores_save_path, obj=df_scores)
            logging.info(f"Best performing model: {best_model_key} successfully saved")

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
