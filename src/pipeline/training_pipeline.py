import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of Trainpipeline class is respobnsible for starting data ingestion componewnt
        """
        try:
            logging.info("Entered the start data ingestion method of Trainpipeline class")
            logging.info("Getting the data from source")
            data_ingestion = DataIngestion(
            data_ingestion_config=self.data_ingestion_config
        )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train set and test set from source")
            logging.info("Exited the start data ingestion method of train pipeline class")

            return data_ingestion_artifact  
    

        except Exception as e:
            raise CustomException(e, sys)
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method is responsible for starting data validation component.
        """
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")

            
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact, 
                data_validation_config=self.data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Completed data validation process")
            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact  
        except Exception as e:
            raise CustomException(e, sys)
    
    def run_pipeline(self):
        try:
            logging.info(">>>>>>> Training Pipeline Started <<<<<<<<<")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            logging.info(">>>>>>> Training Pipeline Completed <<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)