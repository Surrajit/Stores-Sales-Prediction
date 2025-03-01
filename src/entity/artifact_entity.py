from dataclasses import dataclass

@dataclass

class DataIngestionArtifact:
    """
    Artifact class for Data Ingestion outputs
    """
    train_file_path: str
    test_file_path: str
    
@dataclass
class DataValidationArtifact:
    """
    Artifact class for Data Validation outputs
    """
    validation_status: bool
    message: str
    drift_report_file_path: str
    train_file_path: str  
    test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str    

@dataclass
class ModelTrainerArtifact:
    """
    Artifact class for Model Training outputs.
    """
    trained_model_file_path: str
    train_rmse: float
    test_rmse: float
    train_r2: float
    test_r2: float  

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluated_model_path: str
    evaluation_report_file_path: str
    test_rmse: float
    test_mae: float
    test_r2: float
    model_run_id: str

@dataclass
class ModelPusherArtifact:
    deployed_model_path: str
    saved_model_path: str
    model_version: str     