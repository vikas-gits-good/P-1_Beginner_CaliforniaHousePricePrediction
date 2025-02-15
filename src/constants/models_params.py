from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBRFRegressor

models_dict = {
    # Ensemble Models
    "RandomForestRegressor": RandomForestRegressor(
        n_jobs=-1, random_state=44, criterion="friedman_mse"
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        loss="squared_error", criterion="friedman_mse", random_state=45
    ),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=44, loss="square"),
    "XGBRegressor": XGBRegressor(),
    "XGBRFRegressor": XGBRFRegressor(),
    # # Tree Models
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=44),
    # Neightbour Models
    "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1),
    # Linear Models
    "Ridge": Ridge(random_state=44),
    "LinearRegression": LinearRegression(n_jobs=-1),
    "MLPRegressor": MLPRegressor(random_state=44),
}

params_dict = {
    "RandomForestRegressor": {
        #     "n_estimators": [200],  # , 200, 250],
        #     "max_depth": [40],  # [None, 40, 60],
        #     "max_leaf_nodes": [20],  # [None, 20, 30],
    },
    "GradientBoostingRegressor": {
        #     "learning_rate": [0.05],
        #     "n_estimators": [200],
        #     "max_depth": [40],
    },
    "AdaBoostRegressor": {
        #     "learning_rate": [10],
        #     "n_estimators": [200],
    },
    "XGBRegressor": {
        # "": [],
    },
    "XGBRFRegressor": {
        #     "": [],
    },
    "DecisionTreeRegressor": {
        #     "": [],
    },
    "KNeighborsRegressor": {
        # "": [],
    },
    "Ridge": {
        #     "": [],
    },
    "LinearRegression": {
        #     "": [],
    },
    "MLPRegressor": {
        #     "": [],
    },
}
