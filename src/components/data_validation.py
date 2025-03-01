import sys
import json
import pandas as pd
import numpy as np
from pandas import DataFrame  

from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH  
from src.utils.main_utils import read_yaml_file, write_yaml_file 

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initialize data validation with ingestion artifact and validation configuration
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config  
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH) 
        except Exception as e:
            raise CustomException(e, sys)
        
    def validation_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validate the number of columns in the dataframe matches schema
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])  
            logging.info(f"Are all required columns present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e, sys)

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        Validate existence of numerical and categorical columns
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = [
                column for column in self._schema_config["numerical_columns"]
                if column not in dataframe_columns
            ]
            missing_categorical_columns = [
                column for column in self._schema_config["categorical_columns"]  
                if column not in dataframe_columns
            ]

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            if missing_categorical_columns:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            return len(missing_categorical_columns) == 0 and len(missing_numerical_columns) == 0
        except Exception as e:
            raise CustomException(e, sys)  

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Read data from CSV file
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Detect Data Drift using EvidentlyAI (Updated to use `metrics`)
        """
        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(reference_data=reference_df, current_data=current_df)

            report = data_drift_report.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content=json_report)
            n_features = json_report["metrics"][0]["result"]["number_of_columns"]
            n_drifted_features = json_report["metrics"][0]["result"]["number_of_drifted_columns"]
            logging.info(f"{n_drifted_features}/{n_features} features show drift.")
            drift_status = json_report["metrics"][0]["result"]["dataset_drift"]
            return drift_status
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiate data validation process
        """
        try:
            validation_error_msg = ""
            logging.info("Starting Data Validation")

            train_df = self.read_data(file_path=self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            status = self.validation_number_of_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.validation_number_of_columns(dataframe=test_df)
            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += "Columns are missing in testing dataframe. "

            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += "Columns are missing in training dataframe. "

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += "Columns are missing in test dataframe. "

            drift_status = self.detect_dataset_drift(reference_df=train_df, current_df=test_df)

            if drift_status:
                logging.info("Drift detected")
                validation_error_msg += "Drift detected."
            else:
                logging.info("No drift detected")
                validation_error_msg += "No drift detected."

            logging.info(f"Validation error message: {validation_error_msg}")

            data_validation_artifact = DataValidationArtifact(
                validation_status=not bool(validation_error_msg.strip()),
                message=validation_error_msg,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
                train_file_path=self.data_ingestion_artifact.train_file_path,
                test_file_path=self.data_ingestion_artifact.test_file_path  
            )

            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact
    
        except Exception as e:
            raise CustomException(e, sys)





