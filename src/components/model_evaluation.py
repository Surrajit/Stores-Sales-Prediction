import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from urllib.parse import urlparse

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.utils.main_utils import load_object, load_numpy_array_data, write_yaml_file

class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model_path(self) -> str:
        """
        Fetch the path of the best model from MLflow
        """
        try:
            mlflow.set_tracking_uri(self.model_eval_config.mlflow_uri)
            experiment = mlflow.get_experiment_by_name(self.model_eval_config.experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.model_eval_config.experiment_name)
            else:
                experiment_id = experiment.experiment_id

            runs = mlflow.search_runs(experiment_ids=[experiment_id])
            if len(runs) > 0:
                best_run = runs.loc[runs['metrics.test_r2'].idxmax()]
                return os.path.join(best_run['artifact_uri'], "model")
            return None

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics
        """
        try:
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            return {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")

            # Load transformation artifacts
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Load the trained model
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )

            # Initialize MLflow
            mlflow.set_tracking_uri(self.model_eval_config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                # Log model parameters
                if hasattr(trained_model, 'get_params'):
                    mlflow.log_params(trained_model.get_params())

                # Make predictions
                y_pred = trained_model.predict(X_test)
                metrics = self.evaluate_metrics(y_test, y_pred)

                # Log metrics
                mlflow.log_metrics({
                    "test_rmse": metrics["rmse"],
                    "test_mae": metrics["mae"],
                    "test_r2": metrics["r2"]
                })

                # Log model
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(trained_model, "model", registered_model_name="BigMartSalesPredictor")
                else:
                    mlflow.sklearn.log_model(trained_model, "model")

                # Get production model metrics if exists
                production_model_path = self.get_best_model_path()
                production_metrics = None
                is_model_accepted = True

                if production_model_path is not None:
                    production_model = mlflow.sklearn.load_model(production_model_path)
                    y_prod_pred = production_model.predict(X_test)
                    production_metrics = self.evaluate_metrics(y_test, y_prod_pred)
                    
                    # Compare with production model
                    if production_metrics["r2"] >= metrics["r2"]:
                        is_model_accepted = False

                # Save evaluation results
                eval_results = {
                    "model_metrics": metrics,
                    "production_metrics": production_metrics,
                    "is_model_accepted": is_model_accepted
                }
                
                write_yaml_file(
                    file_path=self.model_eval_config.evaluation_report_file_path,
                    content=eval_results
                )

                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    evaluated_model_path=self.model_trainer_artifact.trained_model_file_path,
                    evaluation_report_file_path=self.model_eval_config.evaluation_report_file_path,
                    test_rmse=metrics["rmse"],
                    test_mae=metrics["mae"],
                    test_r2=metrics["r2"],
                    model_run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None
                )

                logging.info(f"Model evaluation completed. Artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys)
        