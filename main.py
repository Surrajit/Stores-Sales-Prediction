from src.logger import logging
from src.exception import CustomException
import sys
from src.pipeline.training_pipeline import TrainPipeline

def main():
    """
    Main function to start the training pipeline
    """
    try:
        logging.info("\n\n>>>>>>> Training Pipeline Started <<<<<<<<")
        
        # Create and run training pipeline
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        
        logging.info(">>>>>>> Training Pipeline Completed <<<<<<<<\n\n")
        
    except Exception as e:
        logging.error(f"Error occurred in main: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(e)
        print(e)