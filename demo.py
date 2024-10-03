
'''
import os
import sys
from src.logger import logging
from src.exception import CustomException



if __name__=="__main__":
    logging.info("Logging has Statrted")

    try:
        a=1/0
    except Exception as e:
        logging.info("Division by Zero")
        raise CustomException(e,sys)
    
    

mongo_db_url = os.getenv("MONGODB_URL")
print(mongo_db_url)


'''
from src.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()

