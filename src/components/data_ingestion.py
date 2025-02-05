import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys
import os

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Method Name : export data into feature store
        Description : This method exports data from source to feature store file

        Output      : Data is returned as an artifact of the data ingestion component
        On Failure  : Write an exception log and then raise an exception 
        """
        try:
            logging.info("Starting data export to feature store")

            # Check if data file exists
            data_file_path = os.path.join("NoteBook","Data", "Train.csv")
            #data_file_path = "NoteBook/Data/Train.csv"
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data File not found at {data_file_path}")

            # Read the data
            logging.info(f"Reading data from {data_file_path}")
            dataframe = pd.read_csv(data_file_path)
            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.data_ingestion_config.raw_data_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)  # Ensure path exists

            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe
        except FileNotFoundError as e:
            logging.error(f"Data File not found: {str(e)}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error(f"Error in export data into feature store: {str(e)}")
            raise CustomException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Method Name : split data as train, test
        Description : This method splits the dataframe into train set and test set based on split ratio

        Output      : Train and test file paths are returned
        On Failure  : Write an exception log and then raise an exception 
        """
        logging.info("Entered split data as train, test method of data ingestion class")
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=0.2,
                random_state=42
            )
            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.train_data_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test file path")
            train_file_path = self.data_ingestion_config.train_data_path
            test_file_path = self.data_ingestion_config.test_data_path

            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)

            return train_file_path, test_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name : initiate data ingestion
        Description : This method initiates the data ingestion components of the training pipeline 

        Output      : Train set and test set are returned as the artifacts of data ingestion components
        On Failure  : Write an exception log and then raise an exception 
        """
        logging.info("Entered initiate data ingestion class")
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from source")

            train_file_path, test_file_path = self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")

            logging.info("Exited initiate data ingestion method of data ingestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
   