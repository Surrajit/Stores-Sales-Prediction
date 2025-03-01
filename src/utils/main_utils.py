import os
import yaml
from src.exception import CustomException
from src.logger import logging
from pathlib import Path
import pandas as pd
import numpy as np 
import pickle
import dill
import sys

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path) as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys)

def write_yaml_file(file_path: str, content: dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
    except Exception as e:
        raise CustomException(e, sys)
    
'''    
def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        if array.size == 0:
            raise ValueError(f"Error: Attempting to save an empty array to {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        np.savez_compressed(file_path, array=array)
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path: str, obj: object):
    try:
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)   

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from an .npz file.
    Parameters:
        file_path (str): Path to the .npz file
    Returns:
        np.ndarray: 2D numpy array
    """
    try:
        logging.info(f"Loading numpy array from file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")

        # Load the npz file
        data = np.load(file_path, allow_pickle=True)
        
        # Print available keys in the npz file for debugging
        logging.info(f"Available keys in npz file: {data.files}")
        
        # Get the array
        array = data['array']
        
        # Log array details
        logging.info(f"Loaded array type: {type(array)}")
        logging.info(f"Loaded array shape: {array.shape}")
        logging.info(f"Loaded array ndim: {array.ndim}")
        
        # Ensure array is 2D
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim == 0:
            raise ValueError(f"Loaded array is 0-dimensional. Check the data transformation output.")
            
        if array.size == 0:
            raise ValueError(f"Loaded array is empty")
            
        logging.info(f"Final array shape: {array.shape}")
        return array
        
    except Exception as e:
        raise CustomException(e, sys)

'''
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array data to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        logging.info(f"Saving array of shape {array.shape} to {file_path}")
        
        # Ensure array is 2D
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim == 0:
            array = np.array([[array]])
            
        np.savez_compressed(file_path, array=array)
        logging.info(f"Successfully saved array at {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file
    """
    try:
        logging.info(f"Loading numpy array from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
            
        data = np.load(file_path, allow_pickle=True)
        array = data['array']
        
        logging.info(f"Loaded array shape: {array.shape}")
        
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim == 0:
            raise ValueError(f"Loaded array is 0-dimensional. Check the data transformation output.")
            
        return array
        
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object by pickling it
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        logging.info(f"Saving object to {file_path}")
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object saved successfully")
        
    except Exception as e:
        raise CustomException(e, sys)      

def load_object(file_path: str) -> object:
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)   

def load_prediction_models():
    global MODEL, PREPROCESSOR
    try:
        import joblib
        api_assets_dir = "api_assets"
        model_path = Path(api_assets_dir) / "model.pkl"
        preprocessor_path = Path(api_assets_dir) / "transformer.pkl"
        
        if not model_path.exists():
            logging.error(f"Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not preprocessor_path.exists():
            logging.error(f"Preprocessor file not found at: {preprocessor_path}")
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        logging.info(f"Attempting to load model from {model_path}")
        MODEL = joblib.load(model_path)
        if MODEL is None:
            logging.error("Model loaded as None")
            raise ValueError("Model loading returned None")
        logging.info("Model loaded successfully")
        
        logging.info(f"Attempting to load preprocessor from {preprocessor_path}")
        PREPROCESSOR = joblib.load(preprocessor_path)
        if PREPROCESSOR is None:
            logging.error("Preprocessor loaded as None")
            raise ValueError("Preprocessor loading returned None")
        logging.info("Preprocessor loaded successfully")
        
        return True
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return False
