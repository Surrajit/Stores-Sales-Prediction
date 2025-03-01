from dataclasses import dataclass
import os
from datetime import datetime
from pathlib import Path

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifact_dir: str = os.path.join("artifacts", TIMESTAMP)

@dataclass

class DataIngestionConfig:
    """
    configuration class for data Ingestion
    """
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    #self.train_test_split_ratio: float = 0.2

@dataclass

class DataValidationConfig:
    """
    Configuration class for data validation
    """
    report_file_path: str = os.path.join("artifacts", "data_validation", "report.yaml")
    drift_report_file_path: str = os.path.join("artifacts", "data_validation", "drift_report.yaml")

@dataclass
class DataTransformationConfig:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.transformed_train_file_path = os.path.join(
            training_pipeline_config.artifact_dir,
            "transformation",
            "transformation_train.npz"
        )
        self.transformed_test_file_path = os.path.join(  
            training_pipeline_config.artifact_dir,
            "transformation",
            "transformation_test.npz"
        )
        self.transformed_object_file_path = os.path.join(  
            training_pipeline_config.artifact_dir,
            "transformation",
            "transformer.pkl"
        )

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for Model Trainer.
    """
    trained_model_file_path: str

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.trained_model_file_path = os.path.join(
            training_pipeline_config.artifact_dir,
            "model_trainer",
            "model.pkl"
        )


@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str = "model_evaluation"
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "BigMart_Sales_Prediction"
    evaluation_report_file_path: str = os.path.join("model_evaluation", "evaluation_report.yaml")

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir = os.path.join(training_pipeline_config.artifact_dir, "model_evaluation")
        self.evaluation_report_file_path = os.path.join(self.model_evaluation_dir, "evaluation_report.yaml")
        os.makedirs(self.model_evaluation_dir, exist_ok=True)

@dataclass
class ModelPusherConfig:
    model_pushing_dir: str
    saved_models_dir: str
    model_promotion_threshold: float        