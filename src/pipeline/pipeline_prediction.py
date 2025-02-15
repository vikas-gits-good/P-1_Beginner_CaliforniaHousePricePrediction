import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Attempting to predict score based on user inputs")

            model_path = "artifacts/04_Models/best_model.pkl"
            preprc_path = "artifacts/03_Pipeline/ppln_prpc.pkl"

            model = load_object(file_path=model_path)
            preprc = load_object(file_path=preprc_path)

            x_pred_tf = preprc.transform(features)
            y_pred = model.predict(x_pred_tf)

            return y_pred

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)


class CustomData:
    def __init__(
        self,
        longitude: float,
        latitude: float,
        housing_median_age: int,
        total_rooms: int,
        total_bedrooms: int,
        population: int,
        households: int,
        median_income: float,
        ocean_proximity: str,
    ):
        self.longitude = float(longitude)
        self.latitude = float(latitude)
        self.housing_median_age = int(housing_median_age)
        self.total_rooms = int(total_rooms)
        self.total_bedrooms = int(total_bedrooms)
        self.population = int(population)
        self.households = int(households)
        self.median_income = float(median_income)
        self.ocean_proximity = str(ocean_proximity)

    def get_DataFrame(self):
        try:
            logging.info("Converting user inputs to DataFrame")
            cust_data = {
                "longitude": [self.longitude],
                "latitude": [self.latitude],
                "housing_median_age": [self.housing_median_age],
                "total_rooms": [self.total_rooms],
                "total_bedrooms": [self.total_bedrooms],
                "population": [self.population],
                "households": [self.households],
                "median_income": [self.median_income],
                "ocean_proximity": [self.ocean_proximity],
            }
            df_cust = pd.DataFrame(cust_data)
            logging.info("Successfully converted user inputs to DataFrame")
            return df_cust

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
