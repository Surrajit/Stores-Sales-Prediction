import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (DataIngestionConfig, 
                                      DataValidationConfig, 
                                      DataTransformationConfig, 
                                      TrainingPipelineConfig,
                                      ModelTrainerConfig,
                                      ModelEvaluationConfig,
                                      ModelPusherConfig)
from src.entity.artifact_entity import (DataIngestionArtifact, 
                                        DataValidationArtifact, 
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact,
                                        ModelEvaluationArtifact,
                                        ModelPusherArtifact)


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
        # Initialize configs
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        self.model_trainer_config = ModelTrainerConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        
        # Update ModelEvaluationConfig initialization to match the class definition
        self.model_evaluation_config = ModelEvaluationConfig(
            training_pipeline_config=self.training_pipeline_config
        )
        
        self.model_pusher_config = ModelPusherConfig(
            model_pushing_dir="deployed_models",
            saved_models_dir="saved_models",
            model_promotion_threshold=0.50  
        )

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of Trainpipeline class is responsible for starting data ingestion component
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
        try:
            logging.info("Starting data validation")
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(
            self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Start data transformation component of training pipeline
        """
        try:
            logging.info("Entered the start data transformation method of Trainpipeline class")
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Performed data Transformation")
            logging.info("Exited the start_data_transformation method of TrainPipeline Class")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)    
        
    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Start model training component of training pipeline
        """
        try:
            logging.info("Entered the start_model_training method of TrainPipeline class")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Completed Model Training")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys) 
            
    def start_model_evaluation(self, model_trainer_artifact: ModelTrainerArtifact, 
                               data_transformation_artifact: DataTransformationArtifact) -> ModelEvaluationArtifact:
        """
        Start model evaluation component of training pipeline
        """
        try:
            logging.info("Starting Model Evaluation")
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact,
                           data_transformation_artifact: DataTransformationArtifact) -> ModelPusherArtifact:
        try:
            logging.info("Starting Model Pusher")
            model_pusher = ModelPusher(
                model_evaluation_artifact=model_evaluation_artifact,
                model_pusher_config=self.model_pusher_config,
                data_transformation_artifact=data_transformation_artifact
            )
            return model_pusher.initiate_model_pushing()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info(">>>>>>> Starting Training Pipeline <<<<<<<<<")
            
            # Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            
            # Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            
            # Model Training
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Model Evaluation with MLflow
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Model Pushing if accepted
            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(
                    model_evaluation_artifact=model_evaluation_artifact,
                    data_transformation_artifact=data_transformation_artifact
                )
                logging.info(f"Model pushed to deployment: {model_pusher_artifact.deployed_model_path}")
            else:
                logging.info("Model did not meet acceptance criteria. Skipping model pushing.")
            
            logging.info(">>>>>>> Training Pipeline Completed <<<<<<<<<")
            
        except Exception as e:
            raise CustomException(e, sys)