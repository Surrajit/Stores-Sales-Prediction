import sys
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from itertools import combinations
from scipy.sparse import issparse

from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils.main_utils import save_object, read_yaml_file, save_numpy_array_data
from src.constants import SCHEMA_FILE_PATH


class FeatureEngineering:
    """
    Custom transfer for feature engineering
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        try:
            df = X.copy()
            
            # Create interaction features for numerical columns
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col1, col2 in combinations(num_cols, 2):
                df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
                df[f"{col1}_{col2}_ratio"] = df[col1] / (df[col2] + 1e-6)  # Adding small value to prevent division by zero
                
            # Create domain-specific features
            if 'Item_Weight' in df.columns and 'Item_Visibility' in df.columns:
                df['Item_Weight_per_Visibility'] = df['Item_Weight'] / (df['Item_Visibility'] + 1e-6)
            
            if 'Outlet_Establishment_Year' in df.columns:
                current_year = 2025  # Update as needed
                df['Outlet_Age'] = current_year - df['Outlet_Establishment_Year']
                
            if 'Item_MRP' in df.columns:
                try:
                    df['Item_MRP_Bucket'] = pd.qcut(df['Item_MRP'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
                except ValueError as e:
                    if "Bin edges must be unique" in str(e):
                        # If all values are the same, assign the middle category
                        df['Item_MRP_Bucket'] = 'M'
                    else:
                        raise e    
                
            return df
        except Exception as e:
            raise CustomException(e,sys)




class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        """
        Initialize DataTransformation with validation artifact and configuration
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("DataTransformation initialized successfully")
            logging.info(f"Schema config loaded: {self._schema_config}")
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Reads data from a CSV file
        """
        try:
            logging.info(f"Reading data from: {file_path}")
            data = pd.read_csv(file_path)
            logging.info(f"Data read successfully with shape: {data.shape}")
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Creates and returns a ColumnTransformer with preprocessing steps
        """
        try:
            logging.info("Creating data transformer object")
            schema_config = self._schema_config
            numerical_columns = schema_config["numerical_columns"]
            categorical_columns = schema_config["categorical_columns"]

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                ]
            )
            feature_engineering = FeatureEngineering()

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            final_pipeline = Pipeline([
                ("feature_engineering", feature_engineering),
                ("preprocessor", preprocessor)
            ])

            logging.info("Data Transformation Pipeline created successfully")
            return final_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation process
        """
        try:
            logging.info("Starting data transformation")

            # Read training and testing data
            train_df = self.read_data(self.data_validation_artifact.train_file_path)
            test_df = self.read_data(self.data_validation_artifact.test_file_path)

            logging.info(f"Train DataFrame Shape: {train_df.shape}")
            logging.info(f"Test DataFrame Shape: {test_df.shape}")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Get target column name
            target_column_name = self._schema_config["target_column"]
            logging.info(f"Target column: {target_column_name}")

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Separated features and target")
            logging.info(f"Input feature train shape: {input_feature_train_df.shape}")
            logging.info(f"Target feature train shape: {target_feature_train_df.shape}")

            # Transform features
            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert target to numpy array and reshape
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            # Convert sparse matrix to dense if necessary
            if issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Combine features and target
            train_arr = np.hstack((input_feature_train_arr, target_feature_train_arr))
            test_arr = np.hstack((input_feature_test_arr, target_feature_test_arr))

            logging.info(f"Final transformed train array shape: {train_arr.shape}")
            logging.info(f"Final transformed test array shape: {test_arr.shape}")

            # Save preprocessing object
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessing_obj
            )

            # Save transformed arrays
            logging.info("Saving transformed arrays")
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # Create artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)