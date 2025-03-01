import sys
import os 
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.utils.main_utils import save_object, load_numpy_array_data
from src.constants import MODEL_TRAINER_MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                model_trainer_config: ModelTrainerConfig):
        """
        Initialize Model Trainer with Transformed data artifacts and model training configuration.
        """
        try:
            logging.info("Initializing ModelTrainer")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            
            # MLflow config - set to use local file system if no server is running
            # This allows tracking without requiring a running MLflow server
            os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
            mlflow.set_experiment("BigMartSalesPredictor")
            
            logging.info("ModelTrainer initialized successfully")
        except Exception as e:
            raise CustomException(e, sys)
            
    def evaluate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Calculate regression metrics"""
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
            
    def get_best_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Train and compare multiple models with optimized configurations
        """
        try:
            # Define models with smaller parameter spaces
            models = {
                'linear': {
                    'model': LinearRegression(),
                    'params': {}  # No hyperparameters to tune
                },
                'lasso': {
                    'model': Lasso(random_state=42),
                    'params': {
                        'alpha': [0.001, 0.01, 0.1, 1.0],
                        'max_iter': [1000]
                    }
                },
                'ridge': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0],
                        'solver': ['auto']
                    }
                },
                'elastic_net': {
                    'model': ElasticNet(random_state=42),
                    'params': {
                        'alpha': [0.001,0.01, 0.1,0.5],
                        'l1_ratio': [0.1, 0.5, 0.9],
                        'max_iter': [10000, 20000]
                    }
                },
                'knn': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': [3, 5, 7],
                        'weights': ['uniform', 'distance']
                    }
                },
                'decision_tree': {
                    'model': DecisionTreeRegressor(random_state=42),
                    'params': {
                        'max_depth': [5, 10, 15],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    }
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100],
                        'max_depth': [8, 12],
                        'min_samples_split': [2, 5],
                        'max_features': ['sqrt']
                    }
                },
                'adaboost': {
                    'model': AdaBoostRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 75],
                        'learning_rate': [0.1, 0.5, 1.0]
                    }
                }
            }
            
            # Add XGBoost and CatBoost only if dataset is not too large
            # You can comment these out if they're still too slow
            if X_train.shape[0] < 50000:  # Only add for smaller datasets
                models.update({
                    'xgboost': {
                        'model': xgb.XGBRegressor(random_state=42),
                        'params': {
                            'n_estimators': [50, 100],
                            'max_depth': [3, 5],
                            'learning_rate': [0.1, 0.2],
                            'subsample': [0.8]
                        }
                    },
                    'catboost': {
                        'model': CatBoostRegressor(verbose=False, random_state=42),
                        'params': {
                            'iterations': [50, 100],
                            'depth': [4, 6],
                            'learning_rate': [0.1, 0.2]
                        }
                    }
                })

            best_score = float('-inf')
            best_model = None
            best_model_name = None
            model_results = {}

            # Determine if we should run quickly or more thoroughly based on data size
            is_small_dataset = X_train.shape[0] < 5000
            cv_folds = 3 if is_small_dataset else 2
            search_iterations = 5 if is_small_dataset else 3
            
            # Start MLflow for tracking all models
            with mlflow.start_run(run_name="model_comparison"):
                for model_name, model_info in models.items():
                    try:
                        logging.info(f"Training {model_name}")
                        
                        # Start a nested run for each model
                        with mlflow.start_run(run_name=model_name, nested=True):
                            # If model has parameters to tune, use RandomizedSearchCV
                            if model_info['params']:
                                search = RandomizedSearchCV(
                                    estimator=model_info['model'],
                                    param_distributions=model_info['params'],
                                    n_iter=search_iterations,
                                    cv=cv_folds,
                                    scoring='r2',
                                    random_state=42,
                                    n_jobs=-1
                                )
                                search.fit(X_train, y_train)
                                model = search.best_estimator_
                                
                                # Log best parameters
                                mlflow.log_params(search.best_params_)
                                logging.info(f"{model_name} best params: {search.best_params_}")
                            else:
                                # For models without hyperparameters, fit directly
                                model = model_info['model']
                                model.fit(X_train, y_train)
                            
                            # Evaluate on train and test data
                            y_train_pred = model.predict(X_train)
                            y_test_pred = model.predict(X_test)
                            
                            # Calculate metrics
                            train_metrics = self.evaluate_metrics(y_train, y_train_pred)
                            test_metrics = self.evaluate_metrics(y_test, y_test_pred)
                            
                            # Log all metrics to MLflow
                            mlflow.log_metrics({
                                "train_rmse": train_metrics["rmse"],
                                "train_mae": train_metrics["mae"],
                                "train_r2": train_metrics["r2"],
                                "test_rmse": test_metrics["rmse"],
                                "test_mae": test_metrics["mae"],
                                "test_r2": test_metrics["r2"]
                            })
                            
                            # Log model to MLflow
                            mlflow.sklearn.log_model(model, "model")
                            
                            # Store result for comparison
                            score = test_metrics["r2"]  
                            logging.info(f"{model_name} test R² score: {score:.4f}")
                        
                        # Store results for each model
                        model_results[model_name] = {
                            'model': model,
                            'score': score
                        }
                        
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_model_name = model_name
                    
                    except Exception as e:
                        logging.warning(f"Error training {model_name}: {e}")
                        continue  # Skip this model and continue with others
                
                # Log all model performances for comparison
                logging.info("Model Performance Summary:")
                for model_name, result in sorted(model_results.items(), key=lambda x: x[1]['score'], reverse=True):
                    logging.info(f"{model_name}: R² = {result['score']:.4f}")
                
                logging.info(f"Best model: {best_model_name} with R² score: {best_score:.4f}")
                
                # Log best model info as a tag in the parent run
                mlflow.set_tag("best_model", best_model_name)
                mlflow.set_tag("best_r2_score", f"{best_score:.4f}")

            return best_model
        except Exception as e:
            raise CustomException(e, sys)        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training pipeline")
            
            # Load and prepare data
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
            
            # Train model with MLflow integration
            model = self.get_best_model(X_train, y_train, X_test, y_test)
            
            # Final evaluation (this was already done in get_best_model)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            logging.info("Final Model Performance Metrics:")
            logging.info(f"Train RMSE: {train_rmse:.2f}")
            logging.info(f"Test RMSE: {test_rmse:.2f}")
            logging.info(f"Train R²: {train_r2:.4f}")
            logging.info(f"Test R²: {test_r2:.4f}")
            
            # Save the model locally as well
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            # Create and return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_rmse=train_rmse,
                test_rmse=test_rmse,
                train_r2=train_r2,
                test_r2=test_r2
            )
            
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException(e, sys)