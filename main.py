from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.model_training import ModelTrainer


def Main():
    try:
        logging.info("Initiating project")

        logging.info("Initiating Data Ingestion")
        di = DataIngestion()
        train_path, test_path = di.initiate_data_ingestion()

        logging.info("Initiating Model Training")
        mt = ModelTrainer(train_path, test_path, best_model_selection_metric="r2_score")
        mt.train_models()

        logging.info("Best model saved and ready for user input prediction")

    except Exception as e:
        logging.info(f"Error: {e}")
        raise CustomException(e)


if __name__ == "__main__":
    Main()
