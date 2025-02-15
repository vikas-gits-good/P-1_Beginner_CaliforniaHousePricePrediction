import os
from dataclasses import dataclass
import pandas as pd

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "01_Data/train.csv")
    test_data_path: str = os.path.join("artifacts", "01_Data/test.csv")
    raw_data_path: str = os.path.join("artifacts", "01_Data/housing.csv")


class DataIngestion:
    def __init__(self):
        self.data_ings_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Entered "initiate_data_ingestion" method')
            self.df = pd.read_csv(self.data_ings_config.raw_data_path)
            self.df["ocean_proximity"] = self.df["ocean_proximity"].replace(
                to_replace={"<1H OCEAN": "Less than 1H OCEAN"}
            )
            logging.info("Read input data successfully")

            os.makedirs(
                os.path.dirname(self.data_ings_config.train_data_path), exist_ok=True
            )

            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(
                self.df, test_size=0.2, random_state=44
            )
            logging.info("Completed train-test split")

            train_set.to_csv(
                self.data_ings_config.train_data_path, index=False, header=True
            )
            logging.info("Saved train set to artifacts")

            test_set.to_csv(
                self.data_ings_config.test_data_path, index=False, header=True
            )
            logging.info("Saved test set to artifacts")
            logging.info("Finished Data Ingestion")

            return (
                self.data_ings_config.train_data_path,
                self.data_ings_config.test_data_path,
            )
        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
