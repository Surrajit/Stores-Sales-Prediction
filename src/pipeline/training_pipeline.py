import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

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
    
    def run_pipeline(self):
        try:
            logging.info(">>>>>>> Training Pipeline Started <<<<<<<<<")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(">>>>>>> Training Pipeline Completed <<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)