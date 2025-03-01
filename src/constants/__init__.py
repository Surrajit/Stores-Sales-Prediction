import os
import sys

ARTIFACTS_DIR = "artifacts"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
TRAIN_FILE_PATH = os.path.join("artifacts", "train.csv")
TEST_FILE_PATH = os.path.join("artifacts", "test.csv")
DRIFT_REPORT_PATH = os.path.join("artifacts", "drift_report.yaml")
MODEL_TRAINER_MODEL_FILE_PATH = os.path.join(ARTIFACTS_DIR, "model_trainer", "model.pkl")