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