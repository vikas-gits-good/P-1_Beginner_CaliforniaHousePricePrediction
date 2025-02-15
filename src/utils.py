import os
import dill
from typing import Literal
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from optuna import ...


def save_object(file_path: str = None, obj=None):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info(f"Error in utils.py/save_object(): {e}")
        raise CustomException(e)


def load_object(file_path: str = None):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info(f"Error in utils.py/load_object(): {e}")
        raise CustomException(e)


def model_scores(
    y_true,
    y_pred: pd.DataFrame = None,
    # score: Literal["All", "r2_score", "mean_square_error"] = "All",
    # model_type:Literal['Classification','Regression']='Regression'
) -> pd.DataFrame:
    scores_dict = {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
        "r2_score": r2_score,
    }
    index_list = list(scores_dict.keys())
    columns_list = list(y_pred.columns)
    df_scores = pd.DataFrame(index=index_list, columns=columns_list)

    for index in index_list:
        for column in columns_list:
            df_scores.loc[index, column] = scores_dict[index](y_true, y_pred[column])

    return df_scores


def evaluate_models(
    x_train=None,
    y_train=None,
    x_test=None,
    y_test=None,
    Models: dict = None,
    Params: dict = None,
    Method: Literal["GridSearchCV", "RandomizedSearchCV", "Optuna"] = "GridSearchCV",
) -> dict:
    try:
        Model_scores = {}
        for i in range(len(list(Models))):
            model_name = list(Models.keys())[i]
            logging.info(f'Started training "{model_name}" model')

            model = list(Models.values())[i]
            param = Params[model_name]

            if Method == "GridSearchCV":
                gs = GridSearchCV(estimator=model, param_grid=param, cv=3, n_jobs=-1)
                gs.fit(x_train, y_train)

            elif Method == "RandomizedSearchCV":
                gs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param,
                    cv=3,
                    n_jobs=-1,
                    n_iter=30,
                    random_state=45,
                )
                gs.fit(x_train, y_train)

            elif Method == "Optuna":
                return None

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train, y_train)
            y_test_pred = model.predict(x_test, y_test)

            score_train = r2_score(y_train, y_train_pred)
            score_test = r2_score(y_test, y_test_pred)

            Model_scores[model_name] = [model, score_train, score_test]
            logging.info(
                f'Finished training "{model_name}" with test set r2_score of: {score_test * 100:.2f}%'
            )
        Model_scores = dict(
            sorted(Model_scores.items(), key=lambda item: item[1][2], reverse=True)
        )
        return Model_scores

    except Exception as e:
        logging.info(f"Error in utils.py/evaluate_models(): {e}")
        raise CustomException(e)
