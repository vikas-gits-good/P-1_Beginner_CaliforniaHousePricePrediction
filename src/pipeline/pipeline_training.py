import os
import pandas as pd
from typing import Literal

from src.logger import logging
from src.exception import CustomException

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    PowerTransformer,
    FunctionTransformer,
    OrdinalEncoder,
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from optuna import


class Save_DataFrame(BaseEstimator, TransformerMixin):
    """Save Dataframe generated during preprocessing and feature engineeering phase to .pkl for analysis

    Args:
        BaseEstimator (object): For pipeline compatibility
        TransformerMixin (object): For pipeline compatibility
    """

    def __init__(self, save_path: str = None):
        self.save_path = save_path

    def fit(self, X, y=None):
        try:
            self.column_names = list(X.columns)
            return self

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def transform(self, X, y=None):
        try:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.column_names)
            if self.save_path:
                # with open(self.save_path, "wb") as file:
                #     pickle.dump(X, file)
                X.to_pickle(self.save_path)
                logging.info(f"DataFrame saved to {self.save_path}")
            return X

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.column_names


class MultiModelEstimator(BaseEstimator, TransformerMixin):
    """Pipeline compatible method to train multiple models

    Args:
        BaseEstimator (object): For pipeline compatibility
        TransformerMixin (object): For pipeline compatibility
    """

    def __init__(
        self,
        models: dict[dict],
        param_grids: dict[dict],
        cv=3,
        scoring: str = None,
        Method: Literal[
            "GridSearchCV", "RandomizedSearchCV", "Optuna"
        ] = "GridSearchCV",
    ):
        self.models = models
        self.param_grids = param_grids
        self.cv = cv
        self.scoring = scoring
        self.method = Method
        self.grid_searches = {}

        if not set(models.keys()).issubset(set(param_grids.keys())):
            missing_params = list(set(models.keys()) - set(param_grids.keys()))
            logging.info("They keys in model dict isnt matching that in params dict")
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params
            )

    def fit(self, X, y=None):
        """Iterates over each model looking for the best hyperparametes

        Args:
            X (array/DataFrame): independant feature
            y (array, optional): dependant feature. Kept for consistency. Defaults to None.

        Raises:
            CustomException: Error during hyperparameter tuning

        Returns:
            self: calculates best paramters and stores it for use with predict method
        """
        try:
            for name, model in self.models.items():
                logging.info(f"Fitting {self.method} for {name}")

                if self.method == "GridSearchCV":
                    gs = GridSearchCV(
                        model,
                        self.param_grids[name],
                        cv=self.cv,
                        scoring=self.scoring,
                        refit=True,
                        n_jobs=-1,
                    )
                    gs.fit(X, y)

                elif self.method == "RandomizedSearchCV":
                    gs = RandomizedSearchCV(
                        model,
                        self.param_grids[name],
                        cv=self.cv,
                        scoring=self.scoring,
                        refit=True,
                        n_jobs=-1,
                    )
                    gs.fit(X, y)

                elif self.method == "Optuna":
                    pass

                self.grid_searches[name] = gs
                logging.info(f"Best parameters for {name}: {gs.best_params_}")

            return self

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def predict(self, X):
        """Iterates over each model and predicts the dependant variable

        Args:
            X (array/DataFrame): independant features from the test/valdiation set

        Raises:
            CustomException: Error during prediction

        Returns:
            Tuple[DataFrame,dict(Models)]: DataFrame with y_pred from each model and a dictionary of models
        """
        try:
            self.predictions_ = {}
            self.models_ = {}

            for name, grid_search in self.grid_searches.items():
                logging.info(f"Predicting {self.method} for {name}")
                best_model = grid_search.best_estimator_
                if hasattr(best_model, "predict_proba"):
                    self.predictions_[name] = best_model.predict_proba(X)
                    self.models_[name] = best_model
                else:
                    self.predictions_[name] = best_model.predict(X)
                    self.models_[name] = best_model

            df_y_pred = pd.DataFrame(self.predictions_)
            return df_y_pred, self.models_

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def get_feature_names_out(self, input_features=None):
        return list(self.predictions_.keys())


class PipelineConstructor:
    """Class that creates a pipeline that will do preprocessing and feature engineering and cant be fit with the model training pipeline"""

    def __init__(
        self,
        cols_numr: list[str] = None,
        cols_catg: list[str] = None,
        cols_drop: list[str] = None,
        catg_ordn: list[list[str]] = None,
    ):
        self.cols_numr = cols_numr
        self.cols_catg = cols_catg
        self.cols_drop = cols_drop
        self.catg_ordn = catg_ordn

    def create_new_feats(self, data: pd.DataFrame = None) -> pd.DataFrame:
        try:
            df = data.copy()
            df["lati/long"] = df["latitude"] / df["longitude"]
            df["total_rooms/house"] = df["total_rooms"] / df["households"]
            df["popl/house"] = df["population"] / df["households"]
            df["bedroom - room"] = df["total_rooms"] - df["total_bedrooms"]
            if self.cols_drop:
                df.drop(columns=self.cols_drop, inplace=True)
            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def CappingOutlier(
        self,
        data,
        columns: list[str] = None,
        threshold: float = 3,
        method: Literal["z_score", "iqr"] = "z_score",
    ):
        try:
            # Temporary workaraound. The dropped columns sent here through self.vols_numr is causing KeyError
            columns = [col for col in data.columns if data[col].dtypes != "O"]
            df = data.copy()
            for col in columns:
                if method == "z_score":
                    mean = df[col].mean()
                    std = df[col].std()
                    upper_bound = mean + threshold * std
                    lower_bound = mean - threshold * std
                elif method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + threshold * IQR
                    lower_bound = Q1 - threshold * IQR

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def create_pipeline(self):
        """Creates a preprocessing + feature engineering pipeline.
        Method is already defined.

        Raises:
            CustomException: Error in pipeline creation

        Returns:
            object: pipeline object to be used with model training
        """
        try:
            # Using OrdinalEncoding
            ppln_prpc = ColumnTransformer(
                transformers=[
                    (
                        "Numerical_Feats",
                        Pipeline(
                            steps=[
                                (
                                    "Features",
                                    FunctionTransformer(func=self.create_new_feats),
                                ),
                                ("Imputer", KNNImputer(n_neighbors=3)),
                                (
                                    "Outlier",
                                    FunctionTransformer(
                                        func=self.CappingOutlier,
                                        kw_args={
                                            "columns": self.cols_numr,
                                            "threshold": 3,
                                            "method": "z_score",
                                        },
                                    ),
                                ),
                                (
                                    "Transformer",
                                    PowerTransformer(
                                        method="yeo-johnson", standardize=True
                                    ),
                                ),
                            ]
                        ),
                        self.cols_numr,
                    ),
                    (
                        "Categorical_Feats",
                        Pipeline(
                            steps=[
                                ("Encoder", OrdinalEncoder(categories=self.catg_ordn)),
                                ("Imputer", KNNImputer(n_neighbors=3)),
                            ]
                        ),
                        self.cols_catg,
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas")

            # Preprocessing -> Feat Engineering
            ppln_fteg = Pipeline(
                steps=[
                    ("ppln_prpc", ppln_prpc),
                    (
                        "Save_DF_pre_proc",
                        Save_DataFrame(
                            save_path=os.path.join(
                                "artifacts/02_DataFrames", "df_prpc.pkl"
                            )
                        ),
                    ),
                    (
                        "Feature_Engineering",
                        PolynomialFeatures(degree=2, include_bias=False),
                    ),
                    ("Feature_Selection", SelectKBest(score_func=f_regression, k=50)),
                    ("Feature_Decomposition", PCA(n_components=30, random_state=42)),
                    (
                        "Save_DF_feat_engn",
                        Save_DataFrame(
                            save_path=os.path.join(
                                "artifacts/02_DataFrames", "df_fteg.pkl"
                            )
                        ),
                    ),
                ]
            ).set_output(transform="pandas")

            return ppln_fteg

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
