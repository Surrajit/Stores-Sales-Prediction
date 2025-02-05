from dataclasses import dataclass

@dataclass

class DataIngestionArtifact:
    """
    Artifact class for Data Ingestion outputs
    """
    train_file_path: str
    test_file_path: str
    