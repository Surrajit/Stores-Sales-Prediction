from dataclasses import dataclass
import os

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


